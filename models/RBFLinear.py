import torch
import torch.nn.functional as F
import math


class RBFLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_centers=10,
        scale_noise=0.1,
        scale_base=1.0,
        scale_rbf=1.0,
        enable_standalone_scale_rbf=True,
        base_activation=torch.nn.SiLU,
        grid_range=[-1, 1],
    ):
        """
        RBF-based activation layer following KAN architecture.
        
        φ(x) = w_b · SiLU(x) + w_r · Σ c_j exp(-(x - μ_j)² / (2σ_j²))
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            num_centers: Number of RBF centers (Q in the formula)
            scale_noise: Initial noise scale for initialization
            scale_base: Scale for base activation weights
            scale_rbf: Scale for RBF weights
            enable_standalone_scale_rbf: Whether to use separate scaling parameter
            base_activation: Base activation function (default: SiLU)
            grid_range: Range for initializing RBF centers
        """
        super(RBFLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        
        # Initialize RBF centers μ_j uniformly distributed in grid_range
        centers = torch.linspace(grid_range[0], grid_range[1], num_centers)
        centers = centers.expand(in_features, -1).contiguous()
        self.register_buffer("centers", centers)
        
        # Learnable parameters
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.rbf_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, num_centers)
        )
        
        # Learnable σ_j (widths of Gaussians)
        self.sigma = torch.nn.Parameter(torch.Tensor(in_features, num_centers))
        
        if enable_standalone_scale_rbf:
            self.rbf_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_rbf = scale_rbf
        self.enable_standalone_scale_rbf = enable_standalone_scale_rbf
        self.base_activation = base_activation()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        with torch.no_grad():
            # Initialize RBF weights with small random values
            noise = (torch.rand(self.out_features, self.in_features, self.num_centers) - 0.5) * self.scale_noise
            self.rbf_weight.data.copy_(
                (self.scale_rbf if not self.enable_standalone_scale_rbf else 1.0) * noise
            )
            
            # Initialize sigma to reasonable values (distance between adjacent centers)
            center_spacing = (self.centers[:, 1] - self.centers[:, 0]).unsqueeze(-1)
            self.sigma.data.fill_(center_spacing.mean().item())
            
            if self.enable_standalone_scale_rbf:
                torch.nn.init.kaiming_uniform_(self.rbf_scaler, a=math.sqrt(5) * self.scale_rbf)
    
    def rbf_basis(self, x: torch.Tensor):
        """
        Compute RBF basis functions: exp(-(x - μ_j)² / (2σ_j²))
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: RBF basis tensor of shape (batch_size, in_features, num_centers)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        # x: (batch_size, in_features) -> (batch_size, in_features, 1)
        x = x.unsqueeze(-1)
        
        # centers: (in_features, num_centers) -> (1, in_features, num_centers)
        centers = self.centers.unsqueeze(0)
        
        # sigma: (in_features, num_centers) -> (1, in_features, num_centers)
        sigma = self.sigma.unsqueeze(0)
        
        # Compute Gaussian RBF: exp(-(x - μ)² / (2σ²))
        # Clamp sigma to avoid division by zero
        sigma_clamped = torch.clamp(sigma, min=1e-6)
        rbf = torch.exp(-((x - centers) ** 2) / (2 * sigma_clamped ** 2))
        
        assert rbf.size() == (x.size(0), self.in_features, self.num_centers)
        return rbf.contiguous()
    
    @property
    def scaled_rbf_weight(self):
        return self.rbf_weight * (
            self.rbf_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_rbf
            else 1.0
        )
    
    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        
        # Base activation output: w_b · SiLU(x)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # RBF output: w_r · Σ c_j · RBF_j(x)
        rbf_output = F.linear(
            self.rbf_basis(x).view(x.size(0), -1),
            self.scaled_rbf_weight.view(self.out_features, -1),
        )
        
        output = base_output + rbf_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute regularization loss for RBF weights.
        Similar to KAN's approach but adapted for RBF.
        """
        l1_fake = self.rbf_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + 1e-10)
        regularization_loss_entropy = -torch.sum(p * p.log() + 1e-10)
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class RBFKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        num_centers=10,
        scale_noise=0.1,
        scale_base=1.0,
        scale_rbf=1.0,
        base_activation=torch.nn.SiLU,
        grid_range=[-1, 1],
    ):
        """
        Full RBF-KAN network with multiple layers.
        
        Args:
            layers_hidden: List of layer sizes [in_features, hidden1, hidden2, ..., out_features]
            num_centers: Number of RBF centers per layer
            scale_noise: Noise scale for initialization
            scale_base: Scale for base weights
            scale_rbf: Scale for RBF weights
            base_activation: Base activation function
            grid_range: Range for RBF centers
        """
        super(RBFKAN, self).__init__()
        self.num_centers = num_centers
        
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                RBFLinear(
                    in_features,
                    out_features,
                    num_centers=num_centers,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_rbf=scale_rbf,
                    base_activation=base_activation,
                    grid_range=grid_range,
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