import jax.numpy as np
from MiniTorch.core.baseclasses import Optimizer, ComputationNode
from MiniTorch.nets.base import Net
from MiniTorch.nets.layers import Linear, PReLU

class SGD(Optimizer):
    def __init__(self, lr, net:Net):
        super().__init__()
        self.lr = lr
        self.net = net
    
    def step(self, ini_grad):
        self.net.backward(ini_grad)
        for layer in self.net.layers:
            if layer.requires_grad and isinstance(layer,ComputationNode):
                    layer.step(self.lr)