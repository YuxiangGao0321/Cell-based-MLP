# Physics informed cell representations for variational formulation of multiscale problems

 - All code for solving multiscale/high-frequency Poisson equations with Physics informed cell representations (cell-based MLP), which is a multiresolution grid model with MLP developed based on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and PyTorch.
   
- All numerical example can be found in `main`.
   
- A tutorial of environment setup will be updated later.

## Abstract
With the rapid advancement of graphical processing units, Physics-Informed Neural Networks (PINNs) are emerging as a promising tool for solving partial differential equations (PDEs). However, PINNs are not well suited for solving PDEs with multiscale features, particularly suffering from slow convergence and poor accuracy. To address this limitation of PINNs, this article proposes physics-informed cell representations for resolving multiscale Poisson problems using a model architecture consisting of multilevel multiresolution grids coupled with a multilayer perceptron (MLP). The grid parameters (i.e., the level-dependent feature vectors) and the MLP parameters (i.e., the weights and biases) are determined using gradient-descent based optimization. The variational (weak) form based loss function accelerates computation by allowing the linear interpolation of feature vectors within grid cells. This cell-based MLP model also facilitates the use of a decoupled training scheme for Dirichlet boundary conditions and a parameter-sharing scheme for periodic boundary conditions, delivering superior accuracy compared to conventional PINNs. Furthermore, the numerical examples highlight improved speed and accuracy in solving PDEs with nonlinear or high-frequency boundary conditions and provide insights into hyperparameter selection. In essence, by cell-based MLP model along with the parallel tiny-cuda-nn library, our implementation improves convergence speed and numerical accuracy.
