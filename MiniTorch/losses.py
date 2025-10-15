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
        from MiniTorch.nets.layers import SoftMax
        self.softmax = SoftMax()
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
        pred = self.softmax.forward(pred)
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
        self.grad_cache['dL_dpred'] = self.softmax.backward(-true / (pred + epsilon))
        return self.grad_cache['dL_dpred']
    
class MSE(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, pred, true):
        self.input = (pred,true)
        batch_size = pred.shape[0]
        loss = np.sum(np.pow(pred-true,2))/batch_size
        return loss
    
    def backward(self, loss = None, make_safe = False):
        pred, true = self.input
        if make_safe:
            pred = np.nan_to_num(pred)
        grad = 2/pred.shape[0] * (pred-true)
        if make_safe:
            grad = np.nan_to_num(grad)
        self.grad_cache['dL_dpred'] = grad
        return self.grad_cache['dL_dpred']
