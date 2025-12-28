# KAN Activation Functions - Comparative Study

Comparison of three Kolmogorov-Arnold Network (KAN) activation functions on Fashion-MNIST.

## 🎯 Activation Functions

| Type | Formula | Key Feature |
|------|---------|-------------|
| **B-Spline** | $\phi(x) = w_{b} \cdot \text{SiLU}(x) + w_{s} \cdot \sum_{i=1}^{G+k} c_i B_i(x)$ | Piecewise polynomials |
| **RBF** | $\phi(x) = w_{b} \cdot \text{SiLU}(x) + w_{r} \cdot \sum_{j=1}^{Q} c_j \exp\left(-\frac{(x - \mu_j)^2}{2\sigma_j^2}\right)$ | Gaussian kernels |
| **Wavelet** | $\phi(x) = w_{b} \cdot \text{SiLU}(x) + w_{w} \cdot \sum_{j=1}^{Q} c_j \psi_{j}(x)$ | Mexican Hat/Morlet |
