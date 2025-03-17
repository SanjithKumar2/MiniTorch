from MiniTorch.core.baseclasses import Loss
import jax.numpy as np

class CCE(Loss):

    def __init__(self):
        super().__init__()

    def loss(self, pred, true,epsilon = 1e-9):
        self.input = (pred,np.array(true))
        loss = -np.mean(np.sum(true * np.log(pred + epsilon),axis=1))
        self.output = loss
        return loss
    def backward(self,loss = None):
        pred, true = self.input
        epsilon = 1e-9
        self.grad_cache['dL_dpred'] = -true / (pred + epsilon)
        return self.grad_cache['dL_dpred']