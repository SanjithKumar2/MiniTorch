from MiniTorch.nets.base import Net

class LRP:

    def __init__(self, net : Net, lrp_params:list, lrp_for_last_activation=False):
        self.net = net
        self.use_last_activ = lrp_for_last_activation
        self.lrp_params = lrp_params
    def propagate_relevance_for_sample(self, R):
        for layer, lrp_param in zip(reversed(self.net.layers), self.lrp_params):
            if isinstance(layer, Linear):
                R = layer.lrp_backward(R,lrp_param["type"], **lrp_param)
                print(R.shape)
            elif isinstance(layer, ReLU):
                R = layer.lrp_backward(R)
        return R