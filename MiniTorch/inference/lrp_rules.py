import jax
import jax.numpy as jnp
from MiniTorch.nets.layers import ReLU, Linear, Conv2D
import warnings
from MiniTorch.inference.utils import _conv_forward, _conv_transpose

#LRP Rules

#Dummy Rule
def _dummy_rule(R_out, **kwargs):
    return R_out

#Linear Layers
#without bias
@jax.jit
def _lrp_lin_backward_0_rule(input, R_out, W, **kwargs):
    z = (input @ W) + 1e-9
    numerator = input.T * W
    R_in = jnp.sum((numerator/z)*R_out,axis=1,keepdims=True).T
    return R_in

@jax.jit
def _lrp_lin_backward_sp_rule(input, R_out, W, w_eps=0.0, z_eps=0.0, **kwargs):
    W = W + w_eps * jnp.maximum(0,W)
    z = input @ W
    z = z + (z_eps * (jnp.mean(z**2)**.5)+ 1e-9)
    numerator = input.T * W
    R_in = jnp.sum((numerator/z)*R_out,axis=1,keepdims=True).T
    return R_in
@jax.jit
def _lrp_lin_backward_zb_rule(input, R_out, W, **kwargs):
    W_max = jnp.maximum(0,W)
    W_min = jnp.minimum(0,W)
    lb = input*0-1
    hb = input*0+1
    z = (input @ W) - (lb @ W_max) - (hb @ W_min) + 1e-9
    numerator = input.T * W - lb.T * W_max - hb.T * W_min
    R_in = jnp.sum((numerator/z)*R_out,axis=1,keepdims=True).T
    return R_in
#with bias
@jax.jit
def _lrp_lin_backward_sp_rule_bias(input, R_out, W, b, w_eps=0.0, z_eps=0.0, **kwargs):
    W = W + w_eps * jnp.maximum(0,W)
    b = b + w_eps * jnp.maximum(0,b)
    z = (input @ W + b) + 1e-9
    z = z + (z_eps * (jnp.mean(z**2)**.5)+ 1e-9)
    numerator = input.T * W
    R_in = jnp.sum((numerator/z)*R_out,axis=1,keepdims=True).T
    return R_in

#LRP for Conv2D
def _lrp_conv_backward_ep_rule(input, R_out, W, z_eps=0.0, stride=(1,1), padding=0,**kwargs):
    z = _conv_forward(input,W,stride,padding)
    z += 1e-12 + z_eps * jnp.sign(z)
    S = R_out/z
    S_w = _conv_transpose(S,W,stride,padding)
    R_in = input * S_w
    return R_in

def _lrp_conv_backward_zb_rule(input, R_out, W, stride=(1,1), padding=0, mean=0., std=1., **kwargs):
    w_p = jnp.maximum(0,W)
    w_n = jnp.minimum(0,W)
    L = jnp.zeros_like(inp_patch) + (0.-mean)/std
    H = jnp.zeros_like(inp_patch) + (1.-mean)/std
    z = _conv_forward(input, W, stride, padding) - _conv_forward(L, w_p,stride, padding) - _conv_forward(H, w_n,stride, padding) + 1e-12
    S = R_out/z
    S_w = _conv_transpose(S, W, stride, padding)
    S_wp = _conv_transpose(S, w_p, stride, padding)
    S_wn = _conv_transpose(S, w_n, stride, padding)
    R_in = input*S_w - L*S_wp - H*S_wn
    return R_in



# def _lrp_conv_backward_zb_rule(input, R_out, W, z_eps=0.0, stride=(1,1),mean=0., std=1., **kwargs):
#     batch, c, h, w = input.shape
#     f, _, kh, kw = W.shape
#     batch, _, oh, ow = R_out.shape
#     stride_h, stride_w = stride
#     R_in = jnp.zeros_like(input)
#     for b in range(batch):
#         for oc in range(f):
#             for i in range(oh):
#                 for j in range(ow):
#                     inp_patch = input[b, :, i*stride_h: (i*stride_h)+kh, j*stride_w:(j*stride_w)+kw]
#                     w = W[oc]
#                     w_p = jnp.maximum(0,w)
#                     w_n = jnp.minimum(0,w)
#                     l_patch = jnp.zeros_like(inp_patch) + (0-mean)/std
#                     h_patch = jnp.zeros_like(inp_patch) + (1-mean)/std
#                     z = jnp.sum(inp_patch * w) - jnp.sum(l_patch * w_p) - jnp.sum(h_patch * w_n)
#                     for k in range(c):
#                         for u in range(kh):
#                             for v in range(kw):
#                                 numen = (inp_patch[k, u, v] - l_patch[k,u,v]) * w_p[k, u, v] - (h_patch[k,u,v] - inp_patch[k,u,v]) * w_n[k, u, v]
#                                 contrib = (numen/z)*R_out[b, oc, i, j]
#                                 R_in = R_in.at[b, k,i*stride_h+u,j*stride_w+v].add(contrib)
#     return R_in

def get_lrp_( layer, rule_type=0, bias=False, **kwargs):
    '''
    Returns the appropriate LRP function or call result #FIX: MAKE EVERYTHING RETURN CALL RESULT

    Parameters:
    layer : ComputationNode -> Any ComputationNode object that supports LRP
    rule_type: int -> the type of lrp rule to use for a layer
            0 -> LRP-0 rule
            1 -> LRP-eps with RMS rule
            2 -> LRP-zb rule for input pixels
            Rules change with each layer so be careful about it
    '''
    if isinstance(layer, Linear):
        if not bias and rule_type==1:
            return _lrp_lin_backward_sp_rule
        elif not bias and rule_type==0:
            return _lrp_lin_backward_0_rule
        elif not bias and rule_type==2:
            return _lrp_lin_backward_zb_rule
        elif bias and rule_type==1:
            return _lrp_lin_backward_sp_rule_bias
        else:
            warnings.warn(f"No such rule type {rule_type} available for layer {type(layer)}")
            return _dummy_rule
    elif isinstance(layer, ReLU):
        pass
    elif isinstance(layer, Conv2D):
        if not bias and rule_type==1:
            return _lrp_conv_backward_ep_rule
        if not bias and rule_type==2:
            return _lrp_conv_backward_ep_rule
    else:
        warnings.warn(f"Rule Not available for the {type(layer)} layer")
        return _dummy_rule
        
#----------------------------------------------------------------------------------------------------------------------------