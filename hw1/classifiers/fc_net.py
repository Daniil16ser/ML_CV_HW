from builtins import object
import os
import numpy as np
from .layers import *
from .layer_utils import *

class TwoLayerNet(object):
    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        self.params = {}
        self.reg = reg

        # Initialize weights and biases
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        N = X.shape[0]
        
        # Forward pass
        out1, cache1 = affine_forward(X, self.params['W1'], self.params['b1'])
        out2, cache2 = relu_forward(out1)
        scores, cache3 = affine_forward(out2, self.params['W2'], self.params['b2'])
        
        if y is None:
            return scores
        
        # Compute loss and gradient
        loss, dscores = softmax_loss(scores, y)
        
        # Add regularization
        loss += 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        
        # Backward pass
        dout2, dw2, db2 = affine_backward(dscores, cache3)
        dout1 = relu_backward(dout2, cache2)
        dx, dw1, db1 = affine_backward(dout1, cache1)
        
        # Add regularization gradients
        dw1 += self.reg * self.params['W1']
        dw2 += self.reg * self.params['W2']
        
        grads = {
            'W1': dw1,
            'b1': db1,
            'W2': dw2,
            'b2': db2
        }
        
        return loss, grads

    def save(self, fname):
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")
    
    def load(self, fname):
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True