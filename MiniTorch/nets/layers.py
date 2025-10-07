from MiniTorch.core.baseclasses import ComputationNode
from MiniTorch.legacy_utils import _conv2d_backward_legacy_v1, _conv2d_forward_legacy_v2,get_kernel_size,get_stride,_conv2d_forward_legacy_v1,_conv_initialize_legacy,_conv2d_backward_legacy_v2, _maxpool2d_backward_legacy_v1,_maxpool2d_forward_legacy_v1
from MiniTorch.inference.lrp_rules import get_lrp_
import numpy as np
from functools import partial
import jax.random as jrandom
import jax.numpy as jnp
import jax
import time
from typing import Literal, List, Tuple, Dict, Any

class Linear(ComputationNode):
    '''
    Represents a linear layer in a neural network.

    Attributes:
    input_size : Size of the input features.
    output_size : Size of the output features.
    initialization : Method for initializing weights.
    parameters : Dictionary containing weights and biases.
    accumulate_grad_norm : Boolean indicating if gradient norms should be accumulated.
    accumulate_parameters : Boolean indicating if parameters should be accumulated.
    '''

    def __init__(self, input_size, output_size,initialization="None", accumulate_grad_norm : bool = False, accumulate_parameters : bool = False, seed_key : int = None, bias=True):
        '''
        Initializes the linear layer with given input and output sizes, and optional initialization method.

        Parameters:
        input_size : Size of the input features.
        output_size : Size of the output features.
        initialization : Method for initializing weights.
        accumulate_grad_norm : Boolean indicating if gradient norms should be accumulated.
        accumulate_parameters : Boolean indicating if parameters should be accumulated.
        seed_key : Optional seed key for random number generation.
        '''
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
        self.bias = bias
    
    def initialize(self, seed_key = False, set_bias = False):
        '''
        Initializes the weights and biases of the linear layer.

        Parameters:
        seed_key : Optional seed key for random number generation.
        set_bias : Boolean indicating if biases should be set.
        '''
        if self.initialization == "xavier":
            limit = jnp.sqrt(6 / (self.input_size + self.output_size))
            self.parameters['W'] = jrandom.uniform(self.seed_key,(self.input_size, self.output_size),minval=-limit,maxval=limit)
        elif self.initialization == "he":
            std = jnp.sqrt(2 / self.input_size)
            self.parameters['W'] = jrandom.normal(self.seed_key,(self.input_size, self.output_size)) * std
        elif self.initialization == "uniform":
            self.parameters['W'] = jrandom.uniform(self.seed_key,(self.input_size, self.output_size),minval=-1/self.input_size,maxval=1/self.output_size)
        else:
            self.parameters['W'] = jrandom.normal(self.seed_key,(self.input_size, self.output_size))
        self.parameters['b'] = jnp.zeros((1,self.output_size))
    @staticmethod
    @jax.jit
    def _linear_forward(input, W, b):
        '''
        Performs the forward pass of the linear layer.

        Parameters:
        input : Input data to the layer.
        W : Weights of the layer.
        b : Biases of the layer.

        Returns:
        Output of the linear transformation.
        '''
        return input @ W + b
    
    @staticmethod
    @jax.jit
    def _linear_backward(input, output_grad, W):
        '''
        Performs the backward pass of the linear layer.

        Parameters:
        input : Input data to the layer.
        output_grad : Gradient of the output.
        W : Weights of the layer.

        Returns:
        Tuple containing gradients with respect to weights, input, and biases.
        '''
        dL_dW = input.T @ output_grad
        dL_dinput = output_grad @ W.T
        dL_db = jnp.sum(output_grad, axis=0, keepdims=True)
        return dL_dW, dL_dinput, dL_db
    
    @staticmethod
    @jax.jit
    def _lrp_backward_0_rule(input, R_out, W):
        z = (input @ W) + 1e-9
        numerator = input.T * W
        R_in = jnp.sum((numerator/z)*R_out,axis=1,keepdims=True).T
        return R_in

    def forward(self, input):
        '''
        Performs a forward pass through the linear layer.

        Parameters:
        input : Input data to the layer.

        Returns:
        Output of the linear transformation.
        '''
        self.input = input
        self.output = self._linear_forward(input, self.parameters['W'], self.parameters['b'])
        return self.output
    
    def backward(self, output_grad):
        self.grad_cache['dL_dW'] ,self.grad_cache['dL_dinput'] ,self.grad_cache['dL_db'] = self._linear_backward(self.input,output_grad,self.parameters['W'])
        return self.grad_cache['dL_dinput']
    def lrp_backward(self, R_out,rule_type="0",bias=False,**kwargs):
        lrp_rule = get_lrp_(self, rule_type, bias)
        R_in = lrp_rule(input = self.input, R_out = R_out, W = self.parameters['W'], b = self.parameters['b'], **kwargs)
        return R_in
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
        if self.bias:
            self.parameters['b'] -= lr * self.grad_cache['dL_db']


class Conv2D(ComputationNode):

    def __init__(self, input_channels : int,kernel_size : int | tuple = 3, no_of_filters = 1, stride = 1, pad = 0, accumulate_grad_norm = False, accumulate_params = False,seed_key = None, bias = True, 
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
            self.parameters['W'], self.parameters['b'] = _conv_initialize_legacy(self.kernel_size,self.no_of_filters,self.input_channels,self.initialization,self.bias)
        else:
            self.initialize(self.seed_key)
    def initialize(self, seed_key):
        fan_in = self.input_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.no_of_filters * self.kernel_size[0] * self.kernel_size[1]
        if self.initialization == "he":
            self.parameters['W'] = jrandom.normal(seed_key, (self.no_of_filters, self.input_channels, self.kernel_size[0], self.kernel_size[1])) * jnp.sqrt(2/fan_in)
        if self.initialization == "xavier":
            std = jnp.sqrt(6/(fan_in + fan_out))
            self.parameters['W'] = jrandom.uniform(seed_key, (self.no_of_filters, self.input_channels, self.kernel_size[0],self.kernel_size[1]),minval = -std, maxval=std)
        else:
            self.parameters['W'] = jrandom.normal(seed_key, (self.no_of_filters, self.input_channels, self.kernel_size[0], self.kernel_size[1]))
        if self.bias:
            self.parameters['b'] = jnp.zeros((self.no_of_filters,))

    @staticmethod
    def _conv2d_forward(X : jax.Array, W : jax.Array,b :jax.Array, stride : tuple, padding: Literal['VALID','SAME'] = 'VALID'):

        # def conv_over_one_batch(X_vec, W_vec, stride, padding):

        #     if X_vec.ndim == 3:
        #         X_vec = X_vec[None,...]
        #     cvout = jax.lax.conv_general_dilated(X_vec,W_vec[None,...],window_strides=stride,padding=padding,
        #                                             dimension_numbers=('NCHW','OIHW','NCHW'))[0,0]
        #     return cvout
        # convout = jax.vmap(jax.vmap(conv_over_one_batch,in_axes=(None,0,None,None)), in_axes=(0,None,None,None))(X,W,stride,padding)
        convout = jax.lax.conv_general_dilated(
        lhs=X, 
        rhs=W, 
        window_strides=stride, 
        padding=padding, 
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        convout += b[None,:,None,None]
        return convout
    @staticmethod
    def _conv2d_backward(X : jax.Array, W : jax.Array, stride : tuple, padding: int, out_grad : jax.Array):
        dL_db = jnp.sum(out_grad, axis=(0,2,3))
        in_channel = X.shape[1]
        batch_size, out_channels, out_h, out_w = out_grad.shape
        kh, kw = W.shape[2], W.shape[3]
        dL_dinput = jnp.zeros_like(X)
        dL_dW = jnp.zeros_like(X)

        input_strided = jax.lax.conv_general_dilated_patches(
            X,
            (kh, kw),
            stride,
            padding='VALID',
            dimension_numbers=('NCHW','OIHW','NCHW')
        )
        input_strided = input_strided.reshape(batch_size,out_h,out_w,in_channel,kh,kw)
        input_strided = input_strided.reshape(batch_size, out_h, out_w, in_channel, kh, kw)
        dL_dW = jnp.einsum('bhwikl,bchw->cikl', input_strided, out_grad, optimize=True)

        out_grad_up = jnp.zeros((batch_size, out_channels, out_h * stride[0], out_w * stride[1]))
        out_grad_up = out_grad_up.at[:, :, ::stride[0], ::stride[1]].set(out_grad)
        out_grad_padded = jnp.pad(out_grad_up, ((0, 0), (0, 0), (padding + 1, padding + 1), (padding + 1, padding + 1)))
        W_rotated = jnp.rot90(W, 2, axes=(2, 3))
        dL_dinput = jnp.einsum('bohw,oikl->bihw', out_grad_padded, W_rotated, optimize=True)

        return dL_dW, dL_db, dL_dinput

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
            output = jax.jit(Conv2D._conv2d_forward, static_argnames=('stride','padding'))(x, W,b, stride)            
        self.output = output
        return self.output
    def backward(self, out_grad):
        dL_dW,dL_db,dL_dinput = None,None,None
        if self.use_legacy_v1:
            dL_dW,dL_db,dL_dinput = _conv2d_backward_legacy_v1(out_grad,self.input,self.kernel_size,self.parameters['W'],self.parameters['b'],self.stride,self.pad)
        elif self.use_legacy_v2:
            dL_dW,dL_db,dL_dinput = _conv2d_backward_legacy_v2(out_grad,self.input,self.kernel_size,self.parameters['W'],self.parameters['b'],self.stride,self.pad)
        else:
            input, W, b, stride,pad =self.input, self.parameters['W'], self.parameters['b'], self.stride, self.pad
            
            if self.pad:
                input = jnp.pad(self.input,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)))
            dL_dW,dL_db,dL_dinput = jax.jit(Conv2D._conv2d_backward,static_argnames=('stride','padding'))(input, W, stride, self.pad, out_grad)
            if self.pad:
                dL_dinput = dL_dinput[:,:,pad:-pad,pad:-pad]

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

class Dropout(ComputationNode):

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.requires_grad = False
        self.mask = None
    def forward(self, x):
        if self.eval:
            return x
        self.input = x
        key = jax.random.PRNGKey(int(time.time()))
        mask = jax.random.bernoulli(key, self.p, x.shape).astype(jnp.float32)
        mask = mask / self.p
        self.mask = mask
        self.output = mask * x
        return self.output
    
    def backward(self, output_grad):
        self.grad_cache['dL_dinput'] = self.mask * output_grad
        return self.grad_cache['dL_dinput']

        
class Flatten(ComputationNode):
    def __init__(self):
        super().__init__()
        self.requires_grad = False
        self.shape = None

    def forward(self,x):
        self.shape = x.shape
        self.input = x
        self.output = jnp.reshape(x,(x.shape[0],-1))
        return self.output
    def backward(self, output_grad):
        dL_dinput= jnp.reshape(output_grad,(self.shape[0],self.shape[1],self.shape[2],self.shape[3]))
        self.grad_cache['dL_dinput']  = dL_dinput
        return dL_dinput
    
class MaxPool2d(ComputationNode):
    def __init__(self, pool_size, pool_stride, use_legacy_v1 = False):
        super().__init__()
        self.pool_size = get_kernel_size(pool_size)
        self.stride = get_stride(pool_stride)
        self.use_legacy_v1 = use_legacy_v1
        self.max_indices = None
        self.requires_grad = False

        
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def _maxpool2d_forward(pool_size, stride, input):
        batch_size, in_channels, in_h, in_w = input.shape
        kh, kw = pool_size
        stride_h, stride_w = stride
        out_h = (in_h - kh) // stride_h + 1
        out_w = (in_w - kw) // stride_w + 1

        input_strided = jax.lax.conv_general_dilated_patches(
            input,
            filter_shape=(kh, kw),
            window_strides=(stride_h, stride_w),
            padding=((0, 0), (0, 0)),
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )  # (batch_size, in_channels, out_h, out_w, kh, kw)
        input_strided = input_strided.reshape(batch_size, in_channels, out_h, out_w, kh, kw)
        # Compute max and argmax over window dims (kh, kw)
        output = jnp.max(input_strided, axis=(4, 5))  # (batch_size, in_channels, out_h, out_w)
        max_indices = jnp.argmax(input_strided.reshape(*input_strided.shape[:4], -1), axis=-1)
        # (batch_size, in_channels, out_h, out_w) - flat indices in kh*kw

        return output, max_indices

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def _maxpool2d_backward(pool_size, stride, input, out_grad, max_indices):
        batch_size, in_channels, in_h, in_w = input.shape
        kh, kw = pool_size
        stride_h, stride_w = stride
        out_h, out_w = out_grad.shape[2], out_grad.shape[3]

        # Convert flat max_indices to 2D offsets within each window
        max_h_offsets = max_indices // kw  # finds the row offset, wehn you flatten in the forward pass the last two dimensions (kh,kw), the max indices range from (0,kw*kh-1), and you divide by kw to ge the row
        max_w_offsets = max_indices % kw   # finds the column offset of each index within a kernel, basically modulo by kw gives the column index within the kernel, like if max_idx  = 5 and k_w = 3 then the row_idx = 5//3 = 1 and col_idx 5%3 = 2

        # Compute input positions where max occurred
        h_starts = jnp.arange(out_h) * stride_h
        w_starts = jnp.arange(out_w) * stride_w
        h_pos = h_starts[None, None, :, None] + max_h_offsets[..., None]  # (b, c, h, w, 1)
        w_pos = w_starts[None, None, :, None] + max_w_offsets[..., None]  # (b, c, h, w, 1)

        # Flatten positions for scattering
        h_pos = h_pos.reshape(-1)
        w_pos = w_pos.reshape(-1)
        batch_idx = jnp.repeat(jnp.arange(batch_size), in_channels * out_h * out_w)
        chan_idx = jnp.tile(jnp.repeat(jnp.arange(in_channels), out_h * out_w), batch_size)
        out_grad_flat = out_grad.reshape(-1)

        # Scatter gradients to dL_dinput
        dL_dinput = jnp.zeros_like(input)
        indices = (batch_idx, chan_idx, h_pos, w_pos)
        dL_dinput = dL_dinput.at[indices].add(out_grad_flat)

        return dL_dinput

    def forward(self, x):
        self.input = x
        if self.use_legacy_v1:
            output = _maxpool2d_forward_legacy_v1(self.pool_size, self.stride, x)
            self.output = output
            self.max_indices = None  # Legacy doesn't cache indices
        else:
            output, max_indices = self._maxpool2d_forward(self.pool_size, self.stride, x)
            self.output = output
            self.max_indices = max_indices
        return output

    def backward(self, output_grad):
        if self.use_legacy_v1:
            dL_dinput = _maxpool2d_backward_legacy_v1(self.pool_size, self.input, output_grad, self.stride)
        else:
            dL_dinput = self._maxpool2d_backward(self.pool_size, self.stride, self.input, output_grad, self.max_indices)
        self.grad_cache = {'dL_input': dL_dinput}  # Assuming ComputationNode expects this
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
    
    def lrp_backward(self, R_out):
        return R_out * (self.input > 0)

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
        self.requires_grad = False
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
    
class Sigmoid(ComputationNode):

    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def sigmoid(self,x):
        return 1/(1+jnp.exp(-x))
    def forward(self, x):
        self.input = x
        self.output = self.sigmoid(x)
        return self.output
    
    def backward(self, output_grad):
        self.grad_cache['dL_dinput'] = (self.output * (1 - self.output)) * output_grad
        return self.grad_cache['dL_dinput']
    
class Tanh(ComputationNode):

    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def tanh(self,x):
        e_pos_x = jnp.exp(x)
        e_neg_x = jnp.exp(-x)
        numerator = e_pos_x - e_neg_x
        denominator = e_neg_x + e_pos_x
        tanh_res = numerator/denominator

    def forward(self, X):
        self.input = X
        self.output = self.tanh(X)
        return self.output
    
    def backward(self, output_grad):
        self.grad_cache['dL_dinput'] = (1 - jnp.power(self.output,2))*output_grad
        return self.grad_cache['dL_dinput']

