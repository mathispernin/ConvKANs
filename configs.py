# =============================================================================
# Training Settings
# =============================================================================
TRAINING = {
    'batch_size': 32,
    'epochs': 30,
    'lr': 1e-3,
    'weight_decay': 1e-4,
}

# =============================================================================
# Activation Configurations
# =============================================================================

BSPLINE = {
    'grid_size': 10,
    'spline_order': 3,
    'scale_noise': 0.1,
    'scale_base': 1.0,
    'scale_spline': 1.0,
    'grid_range': [-1, 1],
}

RBF = {
    'num_centers': 10,
    'scale_noise': 0.1,
    'scale_base': 1.0,
    'scale_rbf': 1.0,
    'grid_range': [-1, 1],
}

WAVELET = {
    'num_wavelets': 10,
    'scale_noise': 0.1,
    'scale_base': 1.0,
    'scale_wav': 1.0,
    'grid_range': [-1, 1],
    'wavelet_type': 'mexican_hat',
}

# =============================================================================
# Experiment Definitions
# =============================================================================

EXPERIMENTS = {
    # Small Convolutional Architecture
    'small_bspline': {
        'name': 'KKAN Small - B-Spline',
        'architecture': 'KKAN_Small',
        'activation': 'spline',
        'config': BSPLINE,
    },
    'small_rbf': {
        'name': 'KKAN Small - RBF',
        'architecture': 'KKAN_Small',
        'activation': 'rbf',
        'config': RBF,
    },
    'small_wavelet': {
        'name': 'KKAN Small - Wavelet',
        'architecture': 'KKAN_Small',
        'activation': 'wavelet',
        'config': WAVELET,
    },
    
    # Medium MLP Architecture
    'medium_bspline': {
        'name': 'Medium KANC MLP - B-Spline',
        'architecture': 'KANC_MLP_Medium',
        'activation': 'spline',
        'config': BSPLINE,
    },
    'medium_rbf': {
        'name': 'Medium KANC MLP - RBF',
        'architecture': 'KANC_MLP_Medium',
        'activation': 'rbf',
        'config': RBF,
    },
    'medium_wavelet': {
        'name': 'Medium KANC MLP - Wavelet',
        'architecture': 'KANC_MLP_Medium',
        'activation': 'wavelet',
        'config': WAVELET,
    },
}
