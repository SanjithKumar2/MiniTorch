import numpy as np
from typing import Tuple

def get_kernel_size(kernel_size):
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    else:
        return kernel_size
def get_stride(stride):
    if isinstance(stride, int):
        return (stride, stride)
    else:
        return stride

def _conv2d_forward_legacy_v2(W, x, stride, b = None, pad = 0):
        no_of_filters, kernel_size_x, kernel_size_y = W.shape
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
        conv_out = np.einsum('bchwkl,fkl->bfhw', x_strided_view, W, optimize=True)
        conv_out += b
        return conv_out

def _conv2d_forward_legacy_v1(W, x, stride, b = None,pad = 0):
        no_of_filters, kernel_size_x, kernel_size_y = W.shape
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

def _conv2d_backward_legacy_v1(out_grad:np.ndarray, input :np.ndarray, kernel_size :Tuple[int], W :np.ndarray, b :np.ndarray, stride : Tuple[int], pad :int) -> np.ndarray:
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
                            np.sum(input[b, :, h_s:h_e, w_s:w_e] * out_grad[b,c,i,j],axis=0)
                        )
            dL_db[c] += np.sum(out_grad[b,c,:,:])
        if pad > 0:
            dL_dinput = dL_dinput[:,:,pad:-pad,pad:-pad]
        return dL_dW, dL_db, dL_dinput

def _conv_initialize_legacy(kernel_size, no_of_fileters, initialization, bias = True):
     W,b = None,None
     if initialization == "he":
          W = np.random.randn(no_of_fileters,kernel_size[0],kernel_size[1]) * (2/kernel_size[0]*kernel_size[1]*no_of_fileters)
     else:
          W = np.random.randn(no_of_fileters,kernel_size[0],kernel_size[1])
     if bias:
          b = np.zeros((no_of_fileters,))
     return W,b