import numpy as np
from typing import Tuple

# Returns the kernel size as a tuple.
# Parameters:
# - kernel_size: An integer or tuple representing the kernel size.
def get_kernel_size(kernel_size):
    '''
    Returns the kernel size as a tuple.

    Parameters:
    kernel_size : An integer or tuple representing the kernel size.
    '''
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    else:
        return kernel_size

# Returns the stride as a tuple.
# Parameters:
# - stride: An integer or tuple representing the stride.
def get_stride(stride):
    '''
    Returns the stride as a tuple.

    Parameters:
    stride : An integer or tuple representing the stride.
    '''
    if isinstance(stride, int):
        return (stride, stride)
    else:
        return stride
    

# Performs the forward pass of a 2D max pooling operation.
# Parameters:
# - pool_size: Size of the pooling window.
# - stride: Stride of the pooling operation.
# - input: Input data to pool.
def _maxpool2d_forward_legacy_v1(pool_size, stride, input):
    '''
    Performs the forward pass of a 2D max pooling operation.

    Parameters:
    pool_size : Size of the pooling window.
    stride : Stride of the pooling operation.
    input : Input data to pool.
    '''
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

# Performs the backward pass of a 2D max pooling operation.
# Parameters:
# - pool_size: Size of the pooling window.
# - input: Input data to pool.
# - out_grad: Gradient of the output.
# - stride: Stride of the pooling operation.
def _maxpool2d_backward_legacy_v1(pool_size, input, out_grad, stride):
    '''
    Performs the backward pass of a 2D max pooling operation.

    Parameters:
    pool_size : Size of the pooling window.
    input : Input data to pool.
    out_grad : Gradient of the output.
    stride : Stride of the pooling operation.
    '''
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

# Performs the forward pass of a 2D convolution operation.
# Parameters:
# - W: Weights of the convolutional layer.
# - x: Input data to convolve.
# - stride: Stride of the convolution.
# - b: Optional bias term.
# - pad: Padding size.
def _conv2d_forward_legacy_v2(W, x, stride, b = None, pad = 0):
    '''
    Performs the forward pass of a 2D convolution operation.

    Parameters:
    W : Weights of the convolutional layer.
    x : Input data to convolve.
    stride : Stride of the convolution.
    b : Optional bias term.
    pad : Padding size.
    '''
    no_of_filters,input_channels, kernel_size_x, kernel_size_y = W.shape
    batch_size, input_channels, input_x, input_y = x.shape
    output_x = (input_x - kernel_size_x)//stride[0] + 1
    output_y = (input_y - kernel_size_y)//stride[1] + 1
    stride_x, stride_y = stride
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2] * stride_x,
        x.strides[3] * stride_y,
        x.strides[2],
        x.strides[3]
    )
    shape = (
        batch_size,
        input_channels,
        output_x,
        output_y,
        kernel_size_x,
        kernel_size_y
    )
    x_strided_view = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    conv_out = np.einsum('bchwkl,fikl->bfhw', x_strided_view, W, optimize=True)
    conv_out += b[None, :, None, None]
    return conv_out

def _conv2d_forward_legacy_v1(W, x, stride, b = None,pad = 0):
        no_of_filters,input_channels, kernel_size_x, kernel_size_y = W.shape
        batch_size, input_channels, input_x, input_y = x.shape
        
        output_x = (input_x - kernel_size_x)//stride[0] + 1
        output_y = (input_y - kernel_size_y)//stride[1] + 1
        out = np.zeros((batch_size, no_of_filters, output_x, output_y))
        for batch in range(batch_size):
            for filter in range(no_of_filters):
                for i in range(output_x):
                    for j in range(output_y):
                        conv_out = np.sum(x[batch, :, i*stride[0]:i*stride[0]+kernel_size_x, j*stride[1]:j*stride[1]+kernel_size_y] * W[filter])
                        out[batch, filter, i, j] = conv_out + b[filter]
        return out

def _conv2d_backward_legacy_v1(out_grad:np.ndarray, input :np.ndarray, kernel_size :Tuple[int], W :np.ndarray, b :np.ndarray, stride : Tuple[int], pad :int = 0) -> np.ndarray:
        batch_size, out_channel, out_h, out_w = out_grad.shape
        dL_dinput = np.zeros_like(input)
        dL_dW = np.zeros_like(W)
        dL_db = np.zeros_like(b)
        dL_dinput = np.pad(dL_dinput,((0,0),(0,0),(pad,pad),(pad,pad)))
        for b in range(batch_size):
            for c in range(out_channel):
                for i in range(out_h):
                    for j in range(out_w):
                        h_s = i * stride[0]
                        h_e = h_s + kernel_size[0]
                        w_s = j * stride[1]
                        w_e = w_s + kernel_size[1]

                        dL_dinput[b, :, h_s:h_e, w_s:w_e] += (
                            W[c] * out_grad[b,c,i,j]
                        )

                        dL_dW[c] += (
                            input[b, :, h_s:h_e, w_s:w_e] * out_grad[b,c,i,j]
                        )
            dL_db[c] += np.sum(out_grad[b,c,:,:])
        if pad > 0:
            dL_dinput = dL_dinput[:,:,pad:-pad,pad:-pad]
        return dL_dW, dL_db, dL_dinput

def _conv2d_backward_legacy_v2(out_grad: np.ndarray, input: np.ndarray, 
                             kernel_size: Tuple[int], W: np.ndarray, 
                             b: np.ndarray, stride: Tuple[int], 
                             pad: int = 0) -> np.ndarray:
    batch_size, out_channel, out_h, out_w = out_grad.shape
    in_channel = input.shape[1]

    dL_dinput = np.zeros_like(input)
    dL_dW = np.zeros_like(W)
    dL_db = np.zeros_like(b)
    if pad > 0:
        dL_dinput_padded = np.zeros((batch_size, in_channel, 
                                   input.shape[2] + 2*pad, 
                                   input.shape[3] + 2*pad))
    else:
        dL_dinput_padded = dL_dinput
    dL_db = np.sum(out_grad, axis=(0, 2, 3))
    kh, kw = kernel_size
    stride_h, stride_w = stride
    input_strided = np.lib.stride_tricks.as_strided(
         input,
         strides = (
              input.strides[0],
              input.strides[2]*stride_h,
              input.strides[3]*stride_w,
              input.strides[1],
              input.strides[2],
              input.strides[3]
         ),shape=(
              batch_size,
              out_h,
              out_w,
              in_channel,
              kh,
              kw) )
    dL_dW = np.einsum('bhwikl,bchw->cikl',input_strided,out_grad,optimize=True)
    out_grad_up = np.zeros((batch_size,out_channel,out_h*stride_h,out_w*stride_w))
    out_grad_up[:,:,::stride_h,::stride_w] = out_grad
    out_grad = np.pad(out_grad_up,((0,0),(0,0),(pad+1,pad+1),(pad+1,pad+1)))
    W = np.rot90(W,2,axes=(2,3))
    dL_dinput = np.einsum('bohw,oikl->bihw',out_grad,W,optimize=True)
    if pad > 0:
        dL_dinput = dL_dinput_padded[:, :, pad:-pad, pad:-pad]
    else:
        dL_dinput = dL_dinput_padded

    return dL_dW, dL_db, dL_dinput


def _conv_initialize_legacy(kernel_size, no_of_fileters,input_channels, initialization, bias = True):
     W,b = None,None
     if initialization == "he":
          W = np.random.randn(no_of_fileters,input_channels,kernel_size[0],kernel_size[1]) * (2/kernel_size[0]*kernel_size[1]*no_of_fileters)
     else:
          W = np.random.randn(no_of_fileters,input_channels,kernel_size[0],kernel_size[1])
     if bias:
          b = np.zeros((no_of_fileters,))
     return W,b