import torch
import numpy as np
from typing import Tuple

def calc_out_dims(matrix: torch.Tensor, kernel_size: int, stride: Tuple[int, int], dilation: Tuple[int, int], padding: Tuple[int, int]):
    """Calcule les dimensions de sortie d'une convolution 2D."""
    batch_size, n_channels, height, width = matrix.shape
    h_out = int(np.floor((height + 2 * padding[0] - kernel_size - (kernel_size - 1) * (dilation[0] - 1)) / stride[0]) + 1)
    w_out = int(np.floor((width + 2 * padding[1] - kernel_size - (kernel_size - 1) * (dilation[1] - 1)) / stride[1]) + 1)
    return h_out, w_out, batch_size, n_channels

def multiple_convs_kan_conv2d(
    matrix: torch.Tensor,
    kernels: list,
    kernel_size: int,
    out_channels: int,
    stride: Tuple[int, int] = (1, 1),
    dilation: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    device: str = "cuda"
) -> torch.Tensor:
    """Effectue une convolution 2D personnalisée avec plusieurs noyaux (KAN)."""
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_size, stride, dilation, padding)
    matrix_out = torch.zeros((batch_size, out_channels, h_out, w_out), device=device) 
    unfold = torch.nn.Unfold((kernel_size, kernel_size), dilation=dilation, padding=padding, stride=stride)
    # Extraire les patches à convolver
    conv_groups = unfold(matrix).view(batch_size, n_channels, kernel_size * kernel_size, h_out * w_out).transpose(2, 3) # Shape: (batch_size, n_channels, num_patches, kernel_size*kernel_size)
    kernels_per_out = len(kernels) // out_channels

    # Ensure kernel submodules (KANLinear) are on the same device as the input
    for kernel in kernels:
        if hasattr(kernel, 'conv') and isinstance(kernel.conv, torch.nn.Module):
            kernel.conv.to(device)

    for c_out in range(out_channels):
        out_accum = torch.zeros((batch_size, h_out, w_out), device=device)
        for k_idx in range(kernels_per_out):
            kernel = kernels[c_out * kernels_per_out + k_idx]
            # On suppose que kernel.conv.forward applique la non-linéarité KAN
            conv_result = kernel.conv.forward(conv_groups[:, k_idx, :, :].flatten(0, 1))
            out_accum += conv_result.view(batch_size, h_out, w_out)
        matrix_out[:, c_out, :, :] = out_accum

    return matrix_out

def add_padding(matrix: np.ndarray, padding: Tuple[int, int]) -> np.ndarray:
    """Ajoute un padding à une matrice 2D numpy."""
    n, m = matrix.shape
    r, c = padding
    padded = np.zeros((n + 2 * r, m + 2 * c))
    padded[r:n + r, c:m + c] = matrix
    return padded