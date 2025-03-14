from MiniTorch.core.baseclasses import ComputationNode
import jax.random as jrandom
import jax.numpy as jnp
import jax
import time

class Linear(ComputationNode):

    def __init__(self, input_size, output_size,initialization="None", accumulate_grad_norm : bool = False, accumulate_parameters : bool = False, seed_key : int = None):
        super().__init__()
        if seed_key == None:
            self.seed_key = jrandom.PRNGKey(int(time.time()))
        else:
            self.seed_key = jrandom.PRNGKey(seed_key)
        self.initialization = initialization
        self.input_size = input_size
        self.output_size = output_size
        self.parameters = {"W":None,"b":None}
        self.initialize()
        self.accumulate_grad_norm = accumulate_grad_norm
        self.accumulate_parameters = accumulate_parameters
    
    def initialize(self, set_bias = False):
        if self.initialization == "xavier":
            limit = jnp.sqrt(6 / (self.input_size + self.output_size))
            self.parameters['W'] = jrandom.uniform(self.seed_key,(self.input_size, self.output_size),minval=-limit,maxval=limit)
        elif self.initialization == "he":
            std = jnp.sqrt(2 / self.input_size)
            self.parameters['W'] = jrandom.normal(self.seed_key,(self.input_size, self.output_size)) * std
        else:
            self.parameters['W'] = jrandom.normal(self.seed_key,(self.input_size, self.output_size))
        self.parameters['b'] = jnp.zeros((1,self.output_size))
    @staticmethod
    @jax.jit
    def __linear_forward(input, W, b):
        return input @ W + b
    
    @staticmethod
    @jax.jit
    def __linear_backward(input, output_grad, W):
        dL_dW = input.T @ output_grad
        dL_dinput = output_grad @ W.T
        dL_db = jnp.sum(output_grad, axis=0, keepdims=True)
        return dL_dW, dL_dinput, dL_db
    
    def forward(self, input):
        self.input = input
        self.output = self.__linear_forward(input, self.parameters['W'], self.parameters['b'])
        return self.output
    
    def backward(self, output_grad):
        self.grad_cache['dL_dW'] ,self.grad_cache['dL_dinput'] ,self.grad_cache['dL_db'] = self.__linear_backward(self.input,output_grad,self.parameters['W'])
        return self.grad_cache['dL_dinput']
    def weights_var_mean(self):
        return self.parameters['W'].var(), self.parameters['W'].mean()
    def bias_var_mean(self):
        return self.parameters['b'].var(), self.parameters['b'].mean()
    
    def step(self, lr):
        if self.accumulate_grad_norm:
            self._accumulate_grad_norm('dL_dW')
            self._accumulate_grad_norm('dL_db')
        if self.accumulate_parameters:
            self._accumulate_parameters('dL_dW', self.weights_var_mean)
        self.parameters['W'] -= lr * self.grad_cache['dL_dW']
        self.parameters['b'] -= lr * self.grad_cache['dL_db']




#-------------------------------------------------------------------------- ACTIVATION LAYERS -------------------------------------------------------------------------------------------------
        
class ReLU(ComputationNode):

    def __init__(self):
        super().__init__()
        self.requires_grad = False
    
    def forward(self,input):
        self.input = input
        self.output = jnp.maximum(0,input)
        return self.output
    
    def backward(self, output_grad):
        dL_dinput = output_grad * (self.input > 0).astype(float)
        self.grad_cache['dL_dinput'] = dL_dinput
        return self.grad_cache['dL_dinput']

class PReLU(ComputationNode):

    def __init__(self,accumulate_grad_norm : bool = False, accumulate_parameters : bool = False):
        super().__init__()
        self.parameters = {'a':jnp.float32(0.25)}
        self.accumulate_grad_norm = accumulate_grad_norm
        self.accumulate_parameters = accumulate_parameters

    def forward(self, input):
        self.input = input
        self.output = jnp.where(input > 0, input, self.parameters['a']*input)
        return self.output
    def backward(self, output_grad):
        self.grad_cache['dL_da'] = jnp.sum(jnp.where(self.input > 0, 0, self.input)*output_grad)
        self.grad_cache['dL_dinput'] = jnp.where(self.input > 0, output_grad, self.parameters['a'] * output_grad)
        return self.grad_cache['dL_dinput']

    def step(self, lr):
        if self.accumulate_grad_norm:
            self._accumulate_grad_norm('dL_da')
        self.parameters['a'] -= lr * self.grad_cache['dL_da']

class SoftMax(ComputationNode):

    def __init__(self, use_legacy_backward = False):
        super().__init__()
        self.use_legacy_backward = use_legacy_backward
    @staticmethod
    @jax.jit
    def __softmax_forward(input):
        inp_exp = jnp.exp(input - jnp.max(input, axis=1, keepdims=True))
        denom = jnp.sum(inp_exp, axis=1, keepdims=True)
        return inp_exp / denom
    @staticmethod
    @jax.jit
    def __softmax_backward(output, output_grad):
        return output * (output_grad - jnp.sum(output * output_grad, axis=1, keepdims=True))
    

    def forward(self, input):
        self.input = input
        self.output = self.__softmax_forward(input)
        return self.output
    
    def legacy_jacobian_softmax(self):
        
        batch_size, num_classes = self.input.shape
        jacobian = jnp.zeros((batch_size,num_classes,num_classes))
        for b in range(batch_size):
            for i in range(num_classes):
                s_i = self.output[b,i]
                for j in range(num_classes):
                    s_j = self.output[b,j]
                    if i == j:
                        jacobian[b, i, j] = s_i * (1 - s_j)
                    else:
                        jacobian[b, i, j] = -1 * s_i * s_j
        return jacobian

    def legacy_jacobian_softmax_v2(self):
        batch_size, classes = self.output.shape
        s = self.output[:,:,None]
        identity = jnp.eye(classes)[None, :, :]
        jacobian = s * identity - jnp.einsum('bij,bij->bij', s, s.transpose(0,2,1))
        return jacobian
    def backward(self, output_grad):
        if self.use_legacy_backward:
            self.grad_cache['dS_dinput'] = self.legacy_jacobian_softmax_v2() 
            self.grad_cache['dL_dinput'] = jnp.einsum('bij,bj->bi', self.grad_cache['dS_dinput'], output_grad)
        self.grad_cache['dL_dinput'] = self.__softmax_backward(self.output,output_grad)
        return self.grad_cache['dL_dinput']