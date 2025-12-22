import torch
import torch.nn.functional as F
import math


class WavLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_wavelets=10,
        scale_noise=0.1,
        scale_base=1.0,
        scale_wav=1.0,
        enable_standalone_scale_wav=True,
        base_activation=torch.nn.SiLU,
        grid_range=[-1, 1],
        wavelet_type='mexican_hat',
    ):
        """
        Wavelet-based activation layer following KAN architecture.
        
        φ(x) = w_b · SiLU(x) + w_w · Σ c_j ψ_j(x)
        
        Mexican Hat wavelet: ψ(x) = (1 - x²) exp(-x²/2)
        where x = (input - μ_j) / s_j
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            num_wavelets: Number of wavelets (Q in the formula)
            scale_noise: Initial noise scale for initialization
            scale_base: Scale for base activation weights
            scale_wav: Scale for wavelet weights
            enable_standalone_scale_wav: Whether to use separate scaling parameter
            base_activation: Base activation function (default: SiLU)
            grid_range: Range for initializing wavelet positions
            wavelet_type: Type of wavelet ('mexican_hat' or 'morlet')
        """
        super(WavLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type
        
        # Initialize wavelet positions μ_j uniformly distributed in grid_range
        positions = torch.linspace(grid_range[0], grid_range[1], num_wavelets)
        positions = positions.expand(in_features, -1).contiguous()
        self.register_buffer("positions", positions)
        
        # Learnable parameters
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.wav_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, num_wavelets)
        )
        
        # Learnable translation (μ_j) and dilation (s_j)
        self.translation = torch.nn.Parameter(torch.Tensor(in_features, num_wavelets))
        self.scale_param = torch.nn.Parameter(torch.Tensor(in_features, num_wavelets))
        
        if enable_standalone_scale_wav:
            self.wav_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_wav = scale_wav
        self.enable_standalone_scale_wav = enable_standalone_scale_wav
        self.base_activation = base_activation()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        with torch.no_grad():
            # Initialize wavelet weights with small random values
            noise = (torch.rand(self.out_features, self.in_features, self.num_wavelets) - 0.5) * self.scale_noise
            self.wav_weight.data.copy_(
                (self.scale_wav if not self.enable_standalone_scale_wav else 1.0) * noise
            )
            
            # Initialize translation to positions
            self.translation.data.copy_(self.positions)
            
            # Initialize scale to reasonable values
            position_spacing = (self.positions[:, 1] - self.positions[:, 0]).unsqueeze(-1)
            self.scale_param.data.fill_(position_spacing.mean().item())
            
            if self.enable_standalone_scale_wav:
                torch.nn.init.kaiming_uniform_(self.wav_scaler, a=math.sqrt(5) * self.scale_wav)
    
    def mexican_hat_wavelet(self, x_normalized):
        """
        Mexican Hat (Ricker) wavelet: ψ(x) = (1 - x²) exp(-x²/2)
        
        Args:
            x_normalized: Normalized input (x - μ) / s
        
        Returns:
            Wavelet response
        """
        x_sq = x_normalized ** 2
        return (1 - x_sq) * torch.exp(-x_sq / 2)
    
    def morlet_wavelet(self, x_normalized):
        """
        Morlet wavelet: ψ(x) = cos(5x) exp(-x²/2)
        
        Args:
            x_normalized: Normalized input (x - μ) / s
            
        Returns:
            Wavelet response
        """
        return torch.cos(5 * x_normalized) * torch.exp(-x_normalized ** 2 / 2)
    
    def wavelet_basis(self, x: torch.Tensor):
        """
        Compute wavelet basis functions.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Wavelet basis tensor of shape (batch_size, in_features, num_wavelets)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        # x: (batch_size, in_features) -> (batch_size, in_features, 1)
        x = x.unsqueeze(-1)
        
        # translation: (in_features, num_wavelets) -> (1, in_features, num_wavelets)
        translation = self.translation.unsqueeze(0)
        
        # scale: (in_features, num_wavelets) -> (1, in_features, num_wavelets)
        scale = self.scale_param.unsqueeze(0)
        
        # Normalize: x_j = (x - μ_j) / s_j
        # Clamp scale to avoid division by zero
        scale_clamped = torch.clamp(scale, min=1e-6)
        x_normalized = (x - translation) / scale_clamped
        
        # Apply wavelet function
        if self.wavelet_type == 'mexican_hat':
            wavelet = self.mexican_hat_wavelet(x_normalized)
        elif self.wavelet_type == 'morlet':
            wavelet = self.morlet_wavelet(x_normalized)
        else:
            raise ValueError(f"Unknown wavelet type: {self.wavelet_type}")
        
        assert wavelet.size() == (x.size(0), self.in_features, self.num_wavelets)
        return wavelet.contiguous()
    
    @property
    def scaled_wav_weight(self):
        return self.wav_weight * (
            self.wav_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_wav
            else 1.0
        )
    
    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        
        # Base activation output: w_b · SiLU(x)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Wavelet output: w_w · Σ c_j · ψ_j(x)
        wav_output = F.linear(
            self.wavelet_basis(x).view(x.size(0), -1),
            self.scaled_wav_weight.view(self.out_features, -1),
        )
        
        output = base_output + wav_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute regularization loss for wavelet weights.
        Similar to KAN's approach but adapted for wavelets.
        """
        l1_fake = self.wav_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + 1e-10)
        regularization_loss_entropy = -torch.sum(p * p.log() + 1e-10)
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class WavKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        num_wavelets=10,
        scale_noise=0.1,
        scale_base=1.0,
        scale_wav=1.0,
        base_activation=torch.nn.SiLU,
        grid_range=[-1, 1],
        wavelet_type='mexican_hat',
    ):
        """
        Full Wavelet-KAN network with multiple layers.
        
        Args:
            layers_hidden: List of layer sizes [in_features, hidden1, hidden2, ..., out_features]
            num_wavelets: Number of wavelets per layer
            scale_noise: Noise scale for initialization
            scale_base: Scale for base weights
            scale_wav: Scale for wavelet weights
            base_activation: Base activation function
            grid_range: Range for wavelet positions
            wavelet_type: Type of wavelet ('mexican_hat' or 'morlet')
        """
        super(WavKAN, self).__init__()
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type
        
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                WavLinear(
                    in_features,
                    out_features,
                    num_wavelets=num_wavelets,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_wav=scale_wav,
                    base_activation=base_activation,
                    grid_range=grid_range,
                    wavelet_type=wavelet_type,
                )
            )
    
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )