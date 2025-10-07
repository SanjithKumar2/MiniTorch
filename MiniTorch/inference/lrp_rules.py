import jax
import jax.numpy as jnp
from MiniTorch.nets.layers import ReLU, Linear
import warnings

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

def get_lrp_( layer, rule_type=0, bias=False):
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
    else:
        warnings.warn(f"Rule Not available for the {type(layer)} layer")
        return _dummy_rule
        
#----------------------------------------------------------------------------------------------------------------------------