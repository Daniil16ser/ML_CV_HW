from builtins import object
import numpy as np

from .layers import *
from .layer_utils import *


class ThreeLayerConvNet(object):
    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        self.params = {}
        self.reg = reg
        self.dtype = dtype


        F, (C, H, W) = num_filters, input_dim
        self.params.update({
          "W1": np.random.normal(loc=0.0, scale=weight_scale, size=(F, C, filter_size, filter_size)),
          "b1": np.zeros((F,)),
          "W2": np.random.normal(loc=0.0, scale=weight_scale, size=( num_filters*(H//2)*(W//2), hidden_dim) ),
          "b2": np.zeros((hidden_dim,)),
          "W3": np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes)),
          "b3": np.zeros((num_classes,))
        })

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None

        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out2, cache2 = affine_relu_forward(out1, W2, b2)
        scores, cache3 = affine_forward(out2, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dout = softmax_loss(scores, y)                     
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)) 

        dout2, dW3, db3 = affine_backward(dout, cache3)
        dout1, dW2, db2 = affine_relu_backward(dout2, cache2)
        dX, dW1, db1 = conv_relu_pool_backward(dout1, cache1)

        grads = {
          "W1": dW1,
          "b1": db1,
          "W2": dW2,
          "b2": db2,
          "W3": dW3,
          "b3": db3
        }

        grads["W1"] += self.reg * W1
        grads["W2"] += self.reg * W2
        grads["W3"] += self.reg * W3

        return loss, grads
