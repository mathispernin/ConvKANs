import torch
import math
from KANLinear import KANLinear
import convolution

class KAN_Convolutional_Layer(torch.nn.Module):
    """
    Couche de convolution KAN avec plusieurs noyaux KAN.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: tuple = (2, 2),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = (-1, 1),
        device: str = "cpu"
    ):
        """
        Args:
            in_channels: Nombre de canaux en entrée.
            out_channels: Nombre de canaux en sortie.
            kernel_size: Taille du noyau (tuple).
            stride: Pas de la convolution.
            padding: Padding.
            dilation: Dilation.
            grid_size: Taille de la grille KAN.
            spline_order: Ordre du spline.
            scale_noise: Échelle du bruit.
            scale_base: Échelle de la base.
            scale_spline: Échelle du spline.
            base_activation: Fonction d'activation de base.
            grid_eps: Epsilon pour la grille.
            grid_range: Plage de la grille.
            device: Appareil utilisé.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.convs = torch.nn.ModuleList([
            KAN_Convolution(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            for _ in range(in_channels * out_channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        return convolution.multiple_convs_kan_conv2d(
            x, self.convs, self.kernel_size[0], self.out_channels,
            self.stride, self.dilation, self.padding, device
        )

class KAN_Convolution(torch.nn.Module):
    """
    Noyau de convolution KAN (utilise KANLinear pour la non-linéarité).
    """
    def __init__(
        self,
        kernel_size: tuple = (2, 2),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = (-1, 1),
        device: str = "cpu"
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.conv = KANLinear(
            in_features=math.prod(kernel_size),
            out_features=1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        return convolution.multiple_convs_kan_conv2d(
            x, [self], self.kernel_size[0], 1,
            self.stride, self.dilation, self.padding, device
        )

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # Si KANLinear a une méthode regularization_loss, on peut l'appeler ici
        return self.conv.regularization_loss(regularize_activation, regularize_entropy)