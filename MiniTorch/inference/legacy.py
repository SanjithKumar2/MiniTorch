import jax
import jax.numpy as jnp

def _lrp_conv_backward_ep_rule(input, R_out, W, z_eps=0.0, stride=(1,1), **kwargs):
    batch, c, h, w = input.shape
    f, _, kh, kw = W.shape
    batch, _, oh, ow = R_out.shape
    stride_h, stride_w = stride
    R_in = jnp.zeros_like(input)

    for b in range(batch):
        for oc in range(f):
            for i in range(oh):
                for j in range(ow):
                    inp_patch = input[b, :, i*stride_h: (i*stride_h)+kh, j*stride_w:(j*stride_w)+kw]
                    w = W[oc]
                    z = jnp.sum(inp_patch * w) + 1e-9
                    z += z_eps * jnp.sign(z)
                    for k in range(c):
                        for u in range(kh):
                            for v in range(kw):
                                numen = inp_patch[k, u, v] * w[k, u, v]
                                contrib = (numen/z)*R_out[b, oc, i, j]
                                R_in = R_in.at[b, k,i*stride_h+u,j*stride_w+v].add(contrib)
    return R_in