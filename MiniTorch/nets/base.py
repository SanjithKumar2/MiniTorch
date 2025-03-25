from typing import List
from MiniTorch.core.baseclasses import ComputationNode, Loss
import jax.random as jrandom
from MiniTorch.nets.layers import ReLU, Linear, Conv2D, PReLU

class Net:
    '''
    Represents a neural network composed of multiple computation layers.

    Attributes:
    layers : List of computation nodes forming the network.
    layer_seed_keys : Dictionary storing seed keys for layer initialization.
    master_key : Key for random number generation to ensure reproducibility.
    '''
    def __init__(self, layers, reproducibility_key = None):
        '''
        Initializes the network with given layers and an optional reproducibility key.

        Parameters:
        layers : List of computation nodes forming the network.
        reproducibility_key : Optional key for ensuring reproducibility.
        '''
        self.layers : List[ComputationNode] = layers
        self.layer_seed_keys = {}
        if reproducibility_key:
            self.master_key = jrandom.PRNGKey(reproducibility_key)
            self.reinitialize_layers()
        
    def __seq(self,layers : list):
        '''
        Sets the sequence of layers in the network.

        Parameters:
        layers : List of computation nodes to set as the network layers.
        '''
        self.layers = layers
    def reinitialize_layers(self):
        '''
        Reinitializes the layers of the network using stored seed keys or generates new ones.
        '''
        if not self.layer_seed_keys:
            self.master_key, key = jrandom.split(self.master_key,2)
            for idx,layer in enumerate(self.layers):
                key, sub_key =jrandom.split(key,2)
                if hasattr(layer, 'initialize'):
                    layer.set_seed(sub_key)
                    layer.initialize(seed_key = sub_key)
                    self.layer_seed_keys[idx] = sub_key
                key = sub_key
        else:
            for idx, key in self.layer_seed_keys.items():
                self.layers[idx].initialize(key)

                
    def forward(self, input):
        '''
        Performs a forward pass through the network.

        Parameters:
        input : Input data to pass through the network.

        Returns:
        Output of the network after the forward pass.
        '''
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    def backward(self,loss):
        '''
        Performs a backward pass through the network to compute gradients.

        Parameters:
        loss : Loss value to backpropagate through the network.

        Returns:
        Gradient after the backward pass.
        '''
        grad = loss
        for layer in self.layers[::-1]:
            if isinstance(layer,ComputationNode) or isinstance(layer, Loss):
                grad = layer.backward(grad)
        return grad
    def print_variance_info(self):
        '''
        Prints variance and mean information for weights, biases, inputs, and outputs of each layer.
        '''
        for idx,layer in enumerate(self.layers):
            if isinstance(layer,Linear) or isinstance(layer,Conv2D):
                print(f"{layer.__class__.__name__} {idx}")
                print(f"Weights Variance and Mean -> {layer.weights_var_mean()}")
                print(f"Bias Variance and Mean -> {layer.bias_var_mean()}")
                print(f"Input Variance and Mean -> {layer.in_var_mean()}")
                print(f"Output Variance and Mean -> {layer.out_var_mean()}")
            if isinstance(layer,ReLU) or isinstance(layer,PReLU):
                print(f"{layer.__class__.__name__} {idx}")
                print(f"Input Variance and Mean -> {layer.in_var_mean()}")
                print(f"Output Variance and Mean -> {layer.out_var_mean()}")
    def print_gradient_info(self):
        n_layers = len(self.layers)
        for idx,layer in enumerate(self.layers[::-1],1):
            if layer.requires_grad:
                if not layer.grad_cache:
                    continue
                print(f"------------------- {layer.__class__.__name__} {n_layers - idx} --------------------")
                for grad_name, grad in layer.grad_cache.items():
                    print(f"Grad {grad_name} Variance and Mean -> {float(grad.var())} , {float(grad.mean())}")
    
    def plot_backprop_grad_dist(self):
        for layer in self.layers[::-1]:
            layer.plot_grad_dist()
