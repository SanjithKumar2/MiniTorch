from MiniTorch.core.baseclasses import Loss
import jax.numpy as np

# Categorical Cross-Entropy (CCE) loss class.
# Inherits from the base Loss class and implements the loss and backward functions.
class CCE(Loss):
    '''
    Categorical Cross-Entropy (CCE) loss class.
    Inherits from the base Loss class and implements the loss and backward functions.
    '''
    # Initializes the CCE loss class.
    def __init__(self):
        '''
        Initializes the CCE loss class.
        '''
        super().__init__()

    # Computes the categorical cross-entropy loss.
    # Parameters:
    # - pred: Predicted probabilities.
    # - true: True labels.
    # - epsilon: Small constant to avoid log(0).
    def loss(self, pred, true,epsilon = 1e-9):
        '''
        Computes the categorical cross-entropy loss.

        Parameters:
        pred : Predicted probabilities.
        true : True labels.
        epsilon : Small constant to avoid log(0).
        '''
        self.input = (pred,np.array(true))
        loss = -np.mean(np.sum(true * np.log(pred + epsilon),axis=1))
        self.output = loss
        return loss

    # Computes the gradient of the loss with respect to the predictions.
    # Parameters:
    # - loss: Optional loss value, not used in this implementation.
    def backward(self,loss = None):
        '''
        Computes the gradient of the loss with respect to the predictions.

        Parameters:
        loss : Optional loss value, not used in this implementation.
        '''
        pred, true = self.input
        epsilon = 1e-9
        self.grad_cache['dL_dpred'] = -true / (pred + epsilon)
        return self.grad_cache['dL_dpred']