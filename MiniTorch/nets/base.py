from typing import List
from MiniTorch.core.baseclasses import ComputationNode, Loss
import jax.random as jrandom
from MiniTorch.nets.layers import ReLU, Linear

class Net:
    def __init__(self, layers, reproducibility_key = None):
        self.layers : List[ComputationNode] = layers
        self.layer_seed_keys = {}
        if reproducibility_key:
            self.master_key = jrandom.PRNGKey(reproducibility_key)
            self.reinitialize_layers()
        
    def __seq(self,layers : list):
        self.layers = layers
    def reinitialize_layers(self):
        if not self.layer_seed_keys:
            self.master_key, key = jrandom.split(self.master_key,2)
            for idx,layer in enumerate(self.layers):
                key, sub_key =jrandom.split(key,2)
                if hasattr(layer, 'initialize'):
                    layer.set_seed(sub_key)
                    layer.initialize()
                    self.layer_seed_keys[idx] = sub_key
                key = sub_key
        else:
            for idx, key in self.layer_seed_keys.items():
                self.layers[idx].initialize()

                
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    def backward(self,loss):
        grad = loss
        for layer in self.layers[::-1]:
            if isinstance(layer,ComputationNode) or isinstance(layer, Loss):
                grad = layer.backward(grad)
        return grad
    def print_variance_info(self):
        for idx,layer in enumerate(self.layers):
            if isinstance(layer,Linear):
                print(f"Linear {idx}")
                print(f"Weights Variance and Mean -> {layer.weights_var_mean()}")
                print(f"Bias Variance and Mean -> {layer.bias_var_mean()}")
                print(f"Input Variance and Mean -> {layer.in_var_mean()}")
                print(f"Output Variance and Mean -> {layer.out_var_mean()}")
            if isinstance(layer,ReLU):
                print(f"Relu {idx}")
                print(f"Input Variance and Mean -> {layer.in_var_mean()}")
                print(f"Output Variance and Mean -> {layer.out_var_mean()}")