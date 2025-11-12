from MiniTorch.core.baseclasses import ComputationNode
from MiniTorch.nets.base import Parameter
from MiniTorch.legacy_utils import _conv2d_backward_legacy_v1, _conv2d_forward_legacy_v2,get_kernel_size,get_stride,_conv2d_forward_legacy_v1,_conv_initialize_legacy,_conv2d_backward_legacy_v2, _maxpool2d_backward_legacy_v1,_maxpool2d_forward_legacy_v1
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

    def __init__(self, input_size, output_size,initialization="he", accumulate_grad_norm : bool = False, accumulate_parameters : bool = False, seed_key : int = None, bias=True):
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
        self.initialize(seed_key=self.seed_key)
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
        self.parameters['W'] = Parameter((self.input_size, self.output_size), self.initialization, seed_key)
        self.parameters['b'] = Parameter((1,self.output_size),is_bias=True)
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
    
    def forward(self, input):
        '''
        Performs a forward pass through the linear layer.

        Parameters:
        input : Input data to the layer.

        Returns:
        Output of the linear transformation.
        '''
        self.input = input
        self.output = self._linear_forward(input, self.parameters['W'].param, self.parameters['b'].param)
        return self.output
    
    def backward(self, output_grad):
        # self.grad_cache['dL_dW'] ,self.grad_cache['dL_dinput'] ,self.grad_cache['dL_db'] = self._linear_backward(self.input,output_grad,self.parameters['W'].param)
        self.parameters['W'].grad ,self.grad_cache['dL_dinput'] ,self.parameters['b'].grad = self._linear_backward(self.input,output_grad,self.parameters['W'].param)
        return self.grad_cache['dL_dinput']
    def lrp_backward(self, R_out,rule_type="0",bias=False,**kwargs):
        from MiniTorch.inference.lrp_rules import get_lrp_
        lrp_rule = get_lrp_(self, rule_type, bias)
        R_in = lrp_rule(input = self.input, R_out = R_out, W = self.parameters['W'], b = self.parameters['b'], **kwargs)
        return R_in
    def weights_var_mean(self):
        return self.parameters['W'].var(), self.parameters['W'].mean()
    def bias_var_mean(self):
        return self.parameters['b'].var(), self.parameters['b'].mean()
    
    def step(self, lr): #FIX : No longer used during updates do something else for grad_norm and param accum if necessary
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
                 initialization = "he", use_legacy_v1 : bool = False, use_legacy_v2:bool = False):
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
        self.parameters['W'] = Parameter((self.no_of_filters, self.input_channels, self.kernel_size[0], self.kernel_size[1]),self.initialization, seed_key)
        if self.bias:
            self.parameters['b'] = Parameter((self.no_of_filters,), is_bias=True)

    @staticmethod
    def _conv2d_forward(X : jax.Array, W : jax.Array,b :jax.Array, stride : tuple, padding: Literal['VALID','SAME'] = 'VALID'):
        convout = jax.lax.conv_general_dilated(
        lhs=X, 
        rhs=jnp.transpose(W, (2, 3, 1, 0)), 
        window_strides=stride, 
        padding=padding, 
        dimension_numbers=('NCHW', 'HWIO', 'NCHW')
        )
        # if bias:
        #     convout += b[None,:,None,None]
        return convout

    @staticmethod
    def _conv2d_backward(X : jax.Array, W : jax.Array, stride : tuple, out_grad : jax.Array, padding: str='VALID'):
        if isinstance(padding, int):
            padding = ((padding, padding), (padding, padding))
        dL_db = jnp.sum(out_grad, axis=(0, 2, 3))
        kh, kw = W.shape[2:]
        input_patches = jax.lax.conv_general_dilated_patches(
            X, (kh, kw), stride, padding
        )
        input_patches = input_patches.reshape(X.shape[0], out_grad.shape[2], out_grad.shape[3], X.shape[1], kh, kw)
        out_grad_reshaped = out_grad.transpose(0, 2, 3, 1)
        dL_dW = jnp.einsum('bhwc,bhwlij->clij', out_grad_reshaped, input_patches, optimize=True)
        dL_dinput = jax.lax.conv_transpose(
            out_grad,
            W,
            strides=stride,
            padding=padding,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
            transpose_kernel=True
        )

        return dL_dW, dL_db, dL_dinput
    def forward(self, x):
        self.input = x
        if self.use_legacy_v1:
            x = np.pad(x,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)))
            self.output = _conv2d_forward_legacy_v1(self.parameters['W'].param, x, self.stride, self.parameters['b'].param)
            return self.output
        if self.use_legacy_v2:
            x = np.pad(x,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)))
            self.output = _conv2d_forward_legacy_v2(self.parameters['W'].param, x, self.stride, self.parameters['b'].param)
            return self.output
        W, b, stride = self.parameters['W'].param, self.parameters['b'].param, self.stride
        with jax.checking_leaks():
            output = jax.jit(Conv2D._conv2d_forward, static_argnames=('stride','padding'))(x, W,b, stride, 'VALID')            
        self.output = output
        return self.output
    def backward(self, out_grad):
        dL_dW,dL_db,dL_dinput = None,None,None
        if self.use_legacy_v1:
            dL_dW,dL_db,dL_dinput = _conv2d_backward_legacy_v1(out_grad,self.input,self.kernel_size,self.parameters['W'].param,self.parameters['b'].param,self.stride,self.pad)
        elif self.use_legacy_v2:
            dL_dW,dL_db,dL_dinput = _conv2d_backward_legacy_v2(out_grad,self.input,self.kernel_size,self.parameters['W'].param,self.parameters['b'].param,self.stride,self.pad)
        else:
            input, W, b, stride,pad =self.input, self.parameters['W'].param, self.parameters['b'].param, self.stride, self.pad
            
            if self.pad:
                input = jnp.pad(self.input,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)))
            dL_dW,dL_db,dL_dinput = jax.jit(Conv2D._conv2d_backward,static_argnames=('stride','padding'))(input, W, stride, out_grad, self.pad if self.pad else "VALID")
            if self.pad and self.pad!='VALID':
                dL_dinput = dL_dinput[:,:,pad:-pad,pad:-pad]

        # self.grad_cache['dL_dW'] = dL_dW
        # self.grad_cache['dL_db'] = dL_db
        self.grad_cache['dL_dinput'] = dL_dinput
        self.parameters['W'].grad = dL_dW
        self.parameters['b'].grad = dL_db
        return dL_dinput
    def lrp_backward(self, R_out,rule_type="0",bias=False,**kwargs):
        from MiniTorch.inference.lrp_rules import get_lrp_
        lrp_rule = get_lrp_(self, rule_type, bias)
        R_in = lrp_rule(input = self.input, R_out = R_out, W = self.parameters['W'], b = self.parameters['b'], **kwargs)
        return R_in
    def weights_var_mean(self):
        return self.parameters['W'].var(), self.parameters['W'].mean()

    def bias_var_mean(self):
        return self.parameters['b'].var(), self.parameters['b'].mean()

    def step(self, lr): #FIX : No longer used during updates do something else for grad_norm and param accum if necessary
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
    def lrp_backward(self, R_out, **kwargs):
        return jnp.reshape(R_out,(self.shape[0],self.shape[1],self.shape[2],self.shape[3]))
    
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

class RNN(ComputationNode):
    def __init__(self, hidden_size, embed_dim, batch_size=1, accumulate_grad_norm=False, accumulate_params=False, initialization="xavier", seed_key=None):
        super().__init__()
        self.h_size = hidden_size
        self.emb_size = embed_dim
        self.batch_size = batch_size
        self.accumulate_grad_norm = accumulate_grad_norm
        self.accumulate_params = accumulate_params
        self.ini = initialization
        if seed_key == None:
            self.seed_key = jrandom.PRNGKey(int(time.time()))
        
        self.parameters = {
            'Wx' : None,
            'Wh' : None,
            'Wy' : None,
            'bh' : None,
            'by' : None
        }
        
        self.tanh = Tanh()
        self.initialize(self.seed_key)

    def initialize(self, seed_key):
        import jax.random as jrandom
        k1, k2, k3 = jrandom.split(seed_key, 3)
        self.parameters['Wx'] = Parameter((self.emb_size, self.h_size), self.ini, k1)
        self.parameters['Wh'] = Parameter((self.h_size, self.h_size), self.ini, k2)
        self.parameters['Wy'] = Parameter((self.h_size, self.emb_size), self.ini, k3)
        self.parameters['bh'] = Parameter((1, self.h_size), initialization=None, seed_key=None, is_bias=True)
        self.parameters['by'] = Parameter((1, self.emb_size), initialization=None, seed_key=None, is_bias=True)

    @staticmethod
    def _rnn_forward(X, h0, Wx, Wh, Wy, bh, by, tanh_forward):
        """
        Static forward pass for jax.jit
        X: (seq_len, batch, emb_dim)
        h0: (batch, hidden_dim)
        """
        seq_len = X.shape[0]
        batch_size = X.shape[1]
        h_states = [h0]
        out_states = []
        inp_states = []
        for t in range(seq_len):
            x_t = X[t]
            H_next = tanh_forward(h_states[-1] @ Wh + x_t @ Wx + bh)
            out_t = H_next @ Wy + by
            h_states.append(H_next)
            out_states.append(out_t)
            inp_states.append(x_t)

        
        h_states = jnp.stack(h_states, axis=0)
        out_states = jnp.stack(out_states, axis=0)
        inp_states =  jnp.stack(inp_states, axis=0)
        return h_states, out_states, inp_states

    def forward(self, X, inference=False):
        """
        X: (batch, seq_len, emb_dim)
        """
        # Clear states
        self.inp_states = []
        self.out_states = []
        self.h_states = [jnp.zeros((self.batch_size, self.h_size))]

        # transpose for time-major
        X_t = jnp.transpose(X, (1, 0, 2))  # (seq_len, batch, emb_dim)
        self.seq_len = int(X_t.shape[0])
        # call jitted static forward
        h_states, out_states, inp_states = self._rnn_forward(
            X_t, self.h_states[0],
            self.parameters['Wx'].param,
            self.parameters['Wh'].param,
            self.parameters['Wy'].param,
            self.parameters['bh'].param,
            self.parameters['by'].param,
            self.tanh.forward
        )

        self.inp_states.extend([inp_states[t] for t in range(inp_states.shape[0])])
        self.h_states.extend([h_states[t] for t in range(h_states.shape[0])])
        self.out_states.extend([out_states[t] for t in range(out_states.shape[0])])

        out_states_t = jnp.transpose(out_states, (1,0,2))  # (batch, seq_len, emb_dim)

        if inference:
            return out_states_t[:, -1, :]
        return out_states_t.reshape(self.batch_size*self.seq_len, self.emb_size)

    @staticmethod
    def _rnn_backward(out_grad, h_states, inp_states, Wx, Wh, Wy, bh, by, tanh_backward):
        """
        Static backward pass for jax.jit
        out_grad: (seq_len, batch, emb_dim)
        h_states: list of h at each step (seq_len, batch, hidden)
        inp_states: list of inputs at each step (seq_len, batch, emb_dim)
        """
        seq_len, batch_size, emb_dim = out_grad.shape
        hidden_dim = h_states[0].shape[1]

        dWx = jnp.zeros((inp_states[0].shape[1], hidden_dim))
        dWh = jnp.zeros((hidden_dim, hidden_dim))
        dWy = jnp.zeros((hidden_dim, emb_dim))
        dbh = jnp.zeros((1, hidden_dim))
        dby = jnp.zeros((1, emb_dim))
        dh_next = jnp.zeros((batch_size, hidden_dim))
        dL_dinput = []
        for t in reversed(range(seq_len)):
            dWy += h_states[t].T @ out_grad[t]
            dby += jnp.sum(out_grad[t], axis=0, keepdims=True)
            dht = out_grad[t] @ Wy.T + dh_next
            dth = tanh_backward(dht)  # h_states[t] pre-activation can be passed if needed
            dWx += inp_states[t].T @ dth
            dWh += h_states[t-1].T @ dth
            dbh += jnp.sum(dth, axis=0, keepdims=True)
            dh_next = dth @ Wh
            dL_dinput.append(dth @ Wx.T)

        grads = {
            'Wx': dWx,
            'Wh': dWh,
            'Wy': dWy,
            'bh': dbh,
            'by': dby
        }
        return grads, jnp.array(dL_dinput)

    def backward(self, out_grad):
        """
        out_grad: (batch*seq_len, emb_dim)
        """
        out_grad = out_grad.reshape(self.batch_size, self.seq_len, self.emb_size)
        out_grad = jnp.transpose(out_grad, (1, 0, 2))  # (seq_len, batch, emb_dim)

        # Call static jitted backward
        grads, dL_dinput = self._rnn_backward(
            out_grad,
            jnp.stack(self.h_states[1:], axis=0),
            jnp.stack(self.inp_states, axis=0),
            self.parameters['Wx'].param,
            self.parameters['Wh'].param,
            self.parameters['Wy'].param,
            self.parameters['bh'].param,
            self.parameters['by'].param,
            self.tanh.backward
        )

        # Assign grads to parameters
        for key in grads:
            self.parameters[key].grad = grads[key]

        return grads, dL_dinput

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
    
    def backward(self, output_grad):
        # if self.use_legacy_backward:
        #     self.grad_cache['dS_dinput'] = self.legacy_jacobian_softmax_v2() 
        #     self.grad_cache['dL_dinput'] = jnp.einsum('bij,bj->bi', self.grad_cache['dS_dinput'], output_grad)
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
        return tanh_res

    def forward(self, X):
        self.input = X
        self.output = self.tanh(X)
        return self.output
    
    def backward(self, output_grad):
        self.grad_cache['dL_dinput'] = (1 - jnp.power(self.output,2))*output_grad
        return self.grad_cache['dL_dinput']

