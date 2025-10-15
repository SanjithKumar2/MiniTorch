import jax.numpy as np
from MiniTorch.core.baseclasses import Optimizer, ComputationNode
from MiniTorch.nets.base import Net
from MiniTorch.nets.layers import Linear, PReLU

# Stochastic Gradient Descent (SGD) optimizer class.
# Inherits from the base Optimizer class and implements the step function for updating network parameters.
class SGD(Optimizer):
    '''
    Stochastic Gradient Descent (SGD) optimizer class.
    Inherits from the base Optimizer class and implements the step function for updating network parameters.
    '''
    # Initializes the SGD optimizer with a learning rate and a neural network.
    # Parameters:
    # - lr: Learning rate for the optimizer.
    # - net: The neural network to optimize.
    def __init__(self, lr, net:Net):
        '''
        Initializes the SGD optimizer with a learning rate and a neural network.

        Parameters:
        lr : Learning rate for the optimizer.
        net : The neural network to optimize.
        '''
        super().__init__()
        self.lr = lr
        self.net = net
    
    # Performs a single optimization step.
    # Parameters:
    # - ini_grad: Initial gradient to start the backward pass.
    def step(self, ini_grad):
        '''
        Performs a single optimization step.

        Parameters:
        ini_grad : Initial gradient to start the backward pass.
        '''
        self.net.backward(ini_grad)
        for layer in self.net.layers:
            if hasattr(layer,"parameters") and isinstance(layer,ComputationNode):
                    for key, val in layer.parameters:
                        if val.requires_grad:
                            layer.parameters[key].param -= self.lr * val.grad

class AdaGrad(Optimizer):

    def __init__(self, lr, net: Net, epsilon=1e-8):
        super().__init__()
        self.lr = lr
        self.net = net
        self.epsilon = epsilon
    
    def step(self, ini_grad):
        self.net.backward(ini_grad)
        for layer in self.net.layers:
            if hasattr(layer, "parameters") and isinstance(layer, ComputationNode):
                for key, val in layer.parameters.items():
                    if val.requires_grad:
                        if not hasattr(val, "sum_sqr_grad"):
                            val.sum_sqr_grad = jnp.zeros_like(val.grad)
                        val.sum_sqr_grad += val.grad ** 2
                        val.param -= (self.lr / (jnp.sqrt(val.sum_sqr_grad) + self.epsilon)) * val.grad

    def clear(self):
        for layer in self.net.layers:
            if hasattr(layer,"parameters") and isinstance(layer,ComputationNode):
                for key, val in layer.parameters.items():
                    if hasattr(val, "sum_sqr_grad"):
                        delattr(val, "sum_sqr_grad")



    
