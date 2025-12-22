# KAN Activation Functions - Comparative Study

Comparison of three Kolmogorov-Arnold Network (KAN) activation functions on Fashion-MNIST.

## 🎯 Activation Functions

| Type | Formula | Key Feature |
|------|---------|-------------|
| **B-Spline** | φ(x) = w_b·SiLU(x) + w_s·Σc_i·B_i(x) | Piecewise polynomials |
| **RBF** | φ(x) = w_b·SiLU(x) + w_r·Σc_j·exp(-(x-μ_j)²/2σ_j²) | Gaussian kernels |
| **Wavelet** | φ(x) = w_b·SiLU(x) + w_w·Σc_j·ψ_j((x-μ_j)/s_j) | Mexican Hat/Morlet |