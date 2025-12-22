import torch
import math
import sys
sys.path.append('./models')
from WavLinear import WavLinear
import convolution


class WavKAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            kernel_size: tuple = (2, 2),
            stride: tuple = (1, 1),
            padding: tuple = (0, 0),
            dilation: tuple = (1, 1),
            num_wavelets: int = 10,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_wav: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_range: tuple = [-1, 1],
            wavelet_type: str = 'mexican_hat',
            device: str = "cpu"
        ):
        """
        Wavelet-KAN Convolutional Layer with multiple convolutions
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (tuple): Size of the kernel
            stride (tuple): Stride of the convolution
            padding (tuple): Padding of the convolution
            dilation (tuple): Dilation of the convolution
            num_wavelets (int): Number of wavelets
            scale_noise (float): Scale of the noise
            scale_base (float): Scale of the base
            scale_wav (float): Scale of the wavelet weights
            base_activation (torch.nn.Module): Activation function
            grid_range (tuple): Range of the wavelet positions
            wavelet_type (str): Type of wavelet ('mexican_hat' or 'morlet')
            device (str): Device to use
        """
        super(WavKAN_Convolutional_Layer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        self.num_wavelets = num_wavelets
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.convs = torch.nn.ModuleList()
        self.stride = stride
        self.wavelet_type = wavelet_type
        
        # Create in_channels * out_channels WavKAN_Convolution objects
        for _ in range(in_channels * out_channels):
            self.convs.append(
                WavKAN_Convolution(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
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
        self.device = x.device
        return convolution.multiple_convs_kan_conv2d(
            x, self.convs, self.kernel_size[0], self.out_channels,
            self.stride, self.dilation, self.padding, self.device
        )


class WavKAN_Convolution(torch.nn.Module):
    def __init__(
            self,
            kernel_size: tuple = (2, 2),
            stride: tuple = (1, 1),
            padding: tuple = (0, 0),
            dilation: tuple = (1, 1),
            num_wavelets: int = 10,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_wav: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_range: tuple = [-1, 1],
            wavelet_type: str = 'mexican_hat',
            device="cpu"
        ):
        """
        Single Wavelet-KAN Convolution unit
        
        Args:
            kernel_size: Size of convolutional kernel
            stride: Convolution stride
            padding: Padding to apply
            dilation: Dilation rate
            num_wavelets: Number of wavelets
            scale_noise: Noise scale for initialization
            scale_base: Base weight scale
            scale_wav: Wavelet weight scale
            base_activation: Base activation function
            grid_range: Range for wavelet positions
            wavelet_type: Type of wavelet
            device: Computing device
        """
        super(WavKAN_Convolution, self).__init__()
        self.num_wavelets = num_wavelets
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.wavelet_type = wavelet_type
        
        self.conv = WavLinear(
            in_features=math.prod(kernel_size),
            out_features=1,
            num_wavelets=num_wavelets,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_wav=scale_wav,
            base_activation=base_activation,
            grid_range=grid_range,
            wavelet_type=wavelet_type,
        )
    
    def forward(self, x: torch.Tensor):
        self.device = x.device
        return convolution.multiple_convs_kan_conv2d(
            x, [self], self.kernel_size[0], 1,
            self.stride, self.dilation, self.padding, self.device
        )
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.conv.regularization_loss(regularize_activation, regularize_entropy)