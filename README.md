# MiniTorch Project

This project is a minimalistic deep learning framework inspired by PyTorch, built using JAX for automatic differentiation and GPU acceleration. It includes core components for building and training neural networks, such as layers, optimizers, and loss functions.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Project Structure

- **`core/`**: Contains base classes for computation nodes, losses, and optimizers.
- **`nets/`**: Contains network and layer implementations.
- **`legacy_utils.py`**: Contains legacy utility functions for operations like convolution and pooling.
- **`plotutils.py`**: Contains utilities for visualizing model outputs.

## Core Components

### ComputationNode

```python
class ComputationNode(abc.ABC):
    """
    Abstract base class for computation nodes in a neural network.

    Attributes:
        input : Input data to the node.
        output : Output data from the node.
        seed : Random seed for reproducibility.
        requires_grad : Boolean indicating if the node requires gradient computation.
        _grad_norm : Dictionary to store gradient norms.
        _accum_params : Dictionary to store accumulated parameters.
        _accum_param_var_mean : Dictionary to store variance and mean of accumulated parameters.
        grad_cache : Dictionary to cache gradients.
        accumulate_grad_norm : Boolean indicating if gradient norms should be accumulated.
        accumulate_parameters : Boolean indicating if parameters should be accumulated.
    """
```

### Loss

```python
class Loss(abc.ABC):
    """
    Abstract base class for loss functions.

    Methods:
        loss(pred, true): Computes the loss between predictions and true labels.
        backward(output_grad): Computes the gradient of the loss with respect to the input.
    """
```

### Optimizer

```python
class Optimizer(abc.ABC):
    """
    Abstract base class for optimizers.

    Methods:
        step(): Performs a single optimization step.
    """
```

## Network Components

### Net

```python
class Net:
    """
    Represents a neural network composed of multiple computation layers.

    Attributes:
        layers : List of computation nodes forming the network.
        layer_seed_keys : Dictionary storing seed keys for layer initialization.
        master_key : Key for random number generation to ensure reproducibility.
    """
```

### Linear

```python
class Linear(ComputationNode):
    """
    Represents a linear layer in a neural network.

    Attributes:
        input_size : Size of the input features.
        output_size : Size of the output features.
        initialization : Method for initializing weights.
        parameters : Dictionary containing weights and biases.
        accumulate_grad_norm : Boolean indicating if gradient norms should be accumulated.
        accumulate_parameters : Boolean indicating if parameters should be accumulated.
    """
```

### Conv2D

```python
class Conv2D(ComputationNode):
    """
    Represents a 2D convolutional layer in a neural network.

    Attributes:
        input_channels : Number of input channels.
        kernel_size : Size of the convolutional kernel.
        no_of_filters : Number of filters in the layer.
        stride : Stride of the convolution.
        pad : Padding size.
        accumulate_grad_norm : Boolean indicating if gradient norms should be accumulated.
        accumulate_params : Boolean indicating if parameters should be accumulated.
        initialization : Method for initializing weights.
        parameters : Dictionary containing weights and biases.
        bias : Boolean indicating if the layer uses a bias term.
    """
```

## Activation Layers

### ReLU

```python
class ReLU(ComputationNode):
    """
    Represents a Rectified Linear Unit (ReLU) activation layer.
    """
```

### PReLU

```python
class PReLU(ComputationNode):
    """
    Represents a Parametric Rectified Linear Unit (PReLU) activation layer.

    Attributes:
        parameters : Dictionary containing the parameter 'a'.
        accumulate_grad_norm : Boolean indicating if gradient norms should be accumulated.
        accumulate_parameters : Boolean indicating if parameters should be accumulated.
    """
```

### SoftMax

```python
class SoftMax(ComputationNode):
    """
    Represents a SoftMax activation layer.

    Attributes:
        use_legacy_backward : Boolean indicating if legacy backward computation should be used.
    """
```

## Loss Functions

### Categorical Cross-Entropy (CCE)

```python
class CCE(Loss):
    """
    Categorical Cross-Entropy (CCE) loss class.
    Inherits from the base Loss class and implements the loss and backward functions.
    """
```

## Optimizers

### Stochastic Gradient Descent (SGD)

```python
class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer class.
    Inherits from the base Optimizer class and implements the step function for updating network parameters.
    """
```


## License

This project is licensed under the MIT License. 