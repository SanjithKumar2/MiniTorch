from MiniTorch.core.baseclasses import ComputationNode
from MiniTorch.legacy_utils import _conv2d_backward_legacy_v1, _conv2d_forward_legacy_v2,get_kernel_size,get_stride,_conv2d_forward_legacy_v1,_conv_initialize_legacy
import numpy as np
import jax.random as jrandom
import jax.numpy as jnp
import jax
import time
from typing import Literal, List, Tuple, Dict, Any

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
    def _linear_forward(input, W, b):
        return input @ W + b
    
    @staticmethod
    @jax.jit
    def _linear_backward(input, output_grad, W):
        dL_dW = input.T @ output_grad
        dL_dinput = output_grad @ W.T
        dL_db = jnp.sum(output_grad, axis=0, keepdims=True)
        return dL_dW, dL_dinput, dL_db
    
    def forward(self, input):
        self.input = input
        self.output = self._linear_forward(input, self.parameters['W'], self.parameters['b'])
        return self.output
    
    def backward(self, output_grad):
        self.grad_cache['dL_dW'] ,self.grad_cache['dL_dinput'] ,self.grad_cache['dL_db'] = self._linear_backward(self.input,output_grad,self.parameters['W'])
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

class Conv2D(ComputationNode):

    def __init__(self, input_channels : int,kernel_size : int | tuple = 3, no_of_filters = 1, stride = 1, pad = None, accumulate_grad_norm = False, accumulate_params = False,seed_key = None, bias = True, 
                 initialization = "None", use_legacy_v1 : bool = False, use_legacy_v2:bool = False):
        super().__init__()
        if seed_key == None:
            self.seed_key = jrandom.PRNGKey(int(time.time()))
        self.kernel_size = get_kernel_size(kernel_size)
        self.input_channels = input_channels
        self.no_of_filters = no_of_filters
        self.stride = get_stride(stride)
        self.pad = pad
        self.accumulate_grad_norm = accumulate_grad_norm
        self.accumulate_params = accumulate_params
        self.initialization = initialization
        self.parameters = {'W': None, 'b': None}
        self.bias = bias

        self.use_legacy_v1 = use_legacy_v1
        self.use_legacy_v2 = use_legacy_v2
        if use_legacy_v1 or use_legacy_v2:
            self.parameters['W'], self.parameters['b'] = _conv_initialize_legacy(self.kernel_size,self.no_of_filters,self.initialization,self.bias)
        else:
            self.initialize(self.seed_key)
    def initialize(self, seed_key):
        if self.initialization == "he":
            self.parameters['W'] = jrandom.normal(seed_key, (self.no_of_filters, self.input_channels, self.kernel_size[0], self.kernel_size[1])) * jnp.sqrt(2/(self.no_of_filters * self.kernel_size[0] * self.kernel_size[1]))
        else:
            self.parameters['W'] = jrandom.normal(seed_key, (self.no_of_filters, self.input_channels, self.kernel_size[0], self.kernel_size[1]))
        if self.bias:
            self.parameters['b'] = jnp.zeros((1,))

    @staticmethod
    def _conv2d_forward(X : jax.Array, W : jax.Array,b : jax.Array, stride : tuple, padding: Literal['VALID','SAME'] = 'VALID'):

        def conv_over_one_batch(X_vec, W_vec, stride, padding):

            if X_vec.ndim == 3:
                X_vec = X_vec[None,...]
            cvout = jax.lax.conv_general_dilated(X_vec,W_vec[None,...],window_strides=stride,padding=padding,
                                                    dimension_numbers=('NCHW','OIHW','NCHW'))[0,0]
            return cvout
        convout = jax.vmap(jax.vmap(conv_over_one_batch,in_axes=(None,0,None,None)), in_axes=(0,None,None,None))(X,W,stride,padding)
        convout += b
        return convout

    def forward(self, x):
        self.input = x
        if self.use_legacy_v1:
            x = np.pad(x,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)))
            self.output = _conv2d_forward_legacy_v1(self.parameters['W'], x, self.stride, self.parameters['b'])
            return self.output
        if self.use_legacy_v2:
            x = np.pad(x,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)))
            self.output = _conv2d_forward_legacy_v2(self.parameters['W'], x, self.stride, self.parameters['b'])
            return self.output
        W, b, stride = self.parameters['W'], self.parameters['b'], self.stride
        with jax.checking_leaks():
            output = jax.jit(Conv2D._conv2d_forward, static_argnames=('stride','padding'))(x, W, b, stride)
        self.output = output
        return self.output
    def backward(self, out_grad):
        dL_dW,dL_db,dL_dinput = None,None,None
        if self.use_legacy_v1 or self.use_legacy_v2:
            dL_dW,dL_db,dL_dinput = _conv2d_backward_legacy_v1(out_grad,self.input,self.kernel_size,self.parameters['W'],self.parameters['b'],self.stride,self.pad)
        self.grad_cache['dL_dW'] = dL_dW
        self.grad_cache['dL_db'] = dL_db
        self.grad_cache['dL_dinput'] = dL_dinput
        return dL_dinput
    
    def weights_var_mean(self):
        return self.parameters['W'].var(), self.parameters['W'].mean()

    def bias_var_mean(self):
        return self.parameters['b'].var(), self.parameters['b'].mean()

    def step(self, lr):
        if self.accumulate_grad_norm:
            self._accumulate_grad_norm('dL_dW')
            self._accumulate_grad_norm('dL_db')
        if self.accumulate_params:
            self._accumulate_parameters('W', self.weights_var_mean)
            self._accumulate_parameters('b', self.bias_var_mean)
        self.parameters['W'] -= lr * self.grad_cache['dL_dW']
        if self.bias:
            self.parameters['b'] -= lr * self.grad_cache['dL_db']

        
class Flatten(ComputationNode):
    def __init__(self):
        super().__init__()
        self.requires_grad = False
        self.shape = None

    def forward(self,x):
        self.shape = x.shape
        self.input = x
        self.output = np.reshape(x,(x.shape[0],-1))
        return self.output
    def backward(self, output_grad):
        dL_dinput= np.reshape(output_grad,(self.shape[0],self.shape[1],self.shape[2],self.shape[3]))
        self.grad_cache['dL_dinput']  = dL_dinput
        return dL_dinput
    
class MaxPool2d(ComputationNode):
    def __init__(self, pool_size, pool_stride, use_legacy_v1 = False):
        super().__init__()
        self.pool_size = get_kernel_size(pool_size)
        self.stride = get_stride(pool_stride)
        self.use_legacy_v1 = use_legacy_v1
    @staticmethod
    def _maxpool2d_forward_legacy_v1(pool_size, stride, input):
        batch_size, input_channels, H, W = input.shape[0],input.shape[1], input.shape[2], input.shape[3]
        output_h = (H - pool_size[0])//stride[0] + 1
        output_w = (W - pool_size[1])//stride[1] + 1
        output = np.zeros((batch_size,input_channels,output_h,output_w))
        for b in range(batch_size):
            for c in range(input_channels):
                for i in range(output_h):
                    for j in range(output_w):
                        h_s = i * stride[0]
                        h_e = h_s + pool_size[0]
                        w_s = j * stride[1]
                        w_e = w_s + pool_size[1]
                        output[b,c,i,j] = np.max(input[b,c,h_s:h_e,w_s:w_e])
        return output
    @staticmethod
    def _maxpool2d_backward_legacy_v1(pool_size, input, out_grad, stride):
        batch_size, input_channels = input.shape[0],input.shape[1]
        out_grad_h, out_grad_w = out_grad.shape[2], out_grad.shape[3]
        dL_dinput = np.zeros_like(input)
        for b in range(batch_size):
            for c in range(input_channels):
                for i in range(out_grad_h):
                    for j in range(out_grad_w):
                        h_s = i * stride[0]
                        h_e = h_s + pool_size[0]
                        w_s = j * stride[1]
                        w_e = w_s + pool_size[1]
                        window = input[b,c,h_s:h_e,w_s:w_e]
                        max_ids = np.unravel_index(np.argmax(window),window.shape)
                        dL_dinput[b,c,h_s + max_ids[0],w_s + max_ids[1]] = out_grad[b,c,i,j]
        return dL_dinput

        
    def forward(self,x):
        self.input = x
        output = None
        if self.use_legacy_v1:
            output = self._maxpool2d_forward_legacy_v1(self.pool_size,self.stride,x)
        self.output = output
        return output

    def backward(self, output_grad):
        dL_dinput = None
        if self.use_legacy_v1:
            dL_dinput = self._maxpool2d_backward_legacy_v1(self.pool_size,self.input,output_grad,self.stride)
        self.grad_cache['dL_input'] = dL_dinput
        return dL_dinput


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
    def _softmax_forward(input):
        inp_exp = jnp.exp(input - jnp.max(input, axis=1, keepdims=True))
        denom = jnp.sum(inp_exp, axis=1, keepdims=True)
        return inp_exp / denom
    @staticmethod
    @jax.jit
    def _softmax_backward(output, output_grad):
        return output * (output_grad - jnp.sum(output * output_grad, axis=1, keepdims=True))
    

    def forward(self, input):
        self.input = input
        self.output = self._softmax_forward(input)
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
        self.grad_cache['dL_dinput'] = self._softmax_backward(self.output,output_grad)
        return self.grad_cache['dL_dinput']