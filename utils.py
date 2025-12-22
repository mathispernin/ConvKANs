from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import sys
sys.path.append('./models')

from models.KANConv import KAN_Convolutional_Layer
from models.RBFKANConv import RBFKAN_Convolutional_Layer
from models.WavKANConv import WavKAN_Convolutional_Layer
from models.KANLinear import KANLinear
from models.RBFLinear import RBFLinear
from models.WavLinear import WavLinear
from architectures import KKAN_Small, KANC_MLP_Medium


def get_fashion_mnist(batch_size=128, num_workers=0, n_train=None, n_test=None, seed=42):
    """Load Fashion-MNIST dataset with optional subsampling and stratification."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_data = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Stratified subsampling for train
    if n_train is not None and n_train < len(train_data):
        targets = np.array(train_data.targets)
        np.random.seed(seed)
        indices = []
        for c in np.unique(targets):
            c_idx = np.where(targets == c)[0]
            n_c = int(n_train * (len(c_idx) / len(targets)))
            indices.extend(np.random.choice(c_idx, n_c, replace=False))
        indices = np.array(indices)
        np.random.shuffle(indices)
        train_data = Subset(train_data, indices)

    # Stratified subsampling for test
    if n_test is not None and n_test < len(test_data):
        targets = np.array(test_data.targets)
        np.random.seed(seed)
        indices = []
        for c in np.unique(targets):
            c_idx = np.where(targets == c)[0]
            n_c = int(n_test * (len(c_idx) / len(targets)))
            indices.extend(np.random.choice(c_idx, n_c, replace=False))
        indices = np.array(indices)
        np.random.shuffle(indices)
        test_data = Subset(test_data, indices)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Loaded Fashion-MNIST with {len(train_data)} training samples and {len(test_data)} test samples.")
    
    return train_loader, test_loader

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_layer_classes(layer_type):
    """Get the appropriate layer classes based on type."""
    if layer_type == 'spline':
        return KAN_Convolutional_Layer, KANLinear
    elif layer_type == 'rbf':
        return RBFKAN_Convolutional_Layer, RBFLinear
    elif layer_type == 'wavelet':
        return WavKAN_Convolutional_Layer, WavLinear
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

def build_model(architecture, activation, config):
    """
    Build a model based on architecture and activation type.
    
    Args:
        architecture: 'KKAN_Small' or 'KANC_MLP'
        activation: 'bspline', 'rbf', or 'wavelet'
        config: Configuration dict for the activation
    
    Returns:
        PyTorch model
    """
    # Select layers based on activation type
    conv_layer, linear_layer = get_layer_classes(activation)
    
    # Build model based on architecture
    if architecture == 'KKAN_Small':
        return KKAN_Small(conv_layer, config, linear_layer, config, num_classes=10)
    elif architecture == 'KANC_MLP_Medium':
        return KANC_MLP_Medium(conv_layer, config, num_classes=10)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
