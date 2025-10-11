import jax
import jax.numpy as jnp

def to_pad_arr(pad):
    if pad==0:
        return 'VALID'
    else:
        return [(pad,pad),(pad,pad)]

def _conv_forward(input, W, stride, padding):
    return jax.lax.conv_general_dilated(
        lhs=input,
        rhs=jnp.transpose(W, (2,3,1,0)),
        window_strides = stride,
        padding=padding if isinstance(padding, str) else to_pad_arr(padding),
        dimension_numbers=('NCHW','HWIO','NCHW')
    )

def _conv_transpose(input, W, stride, padding):
    return jax.lax.conv_transpose(
        lhs = input,
        rhs = W,
        strides = stride,
        padding=padding if isinstance(padding, str) else to_pad_arr(padding),
        dimension_numbers=('NCHW','IOHW','NCHW')
    )