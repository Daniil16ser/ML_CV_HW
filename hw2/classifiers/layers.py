from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """Forward pass for affine (fully connected) layer."""
    x_reshaped = x.reshape(x.shape[0], -1)
    out = x_reshaped @ w + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """Backward pass for affine layer."""
    x, w, b = cache
    x_flat = x.reshape(x.shape[0], -1)
    
    dw = x_flat.T @ dout
    dx_flat = dout @ w.T
    dx = dx_flat.reshape(x.shape)
    db = np.sum(dout, axis=0)
    
    return dx, dw, db

def relu_forward(x):
    """Forward pass for ReLU activation."""
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """Backward pass for ReLU activation."""
    x = cache
    dx = dout * (x > 0)
    return dx

def softmax_loss(x, y):
    """Softmax loss function."""
    N = x.shape[0]
    
    # Shift for numerical stability
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    
    # Compute softmax probabilities
    exp_x = np.exp(shifted_x)
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # Compute loss
    correct_log_probs = -np.log(probs[np.arange(N), y])
    loss = np.sum(correct_log_probs) / N
    
    # Compute gradient
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx = dx / N
    
    return loss, dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization."""
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        cache = (x, x_norm, sample_mean, sample_var, gamma, beta, eps)
        
    elif mode == "test":
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        cache = (x, x_norm, running_mean, running_var, gamma, beta, eps)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization."""
    x, x_norm, mu, var, gamma, beta, eps = cache
    N, D = x.shape
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    
    dx_norm = dout * gamma
    
    dvar = np.sum(dx_norm * (x - mu) * (-0.5) * (var + eps)**(-1.5), axis=0)
    dmu = np.sum(dx_norm * (-1) / np.sqrt(var + eps), axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
    
    dx = dx_norm / np.sqrt(var + eps) + dvar * (2 * (x - mu) / N) + dmu / N
    
    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization."""
    x, x_norm, mu, var, gamma, beta, eps = cache
    N, D = x.shape
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    
    dx_norm = dout * gamma
    
    dx = (1. / N) * (1. / np.sqrt(var + eps)) * (N * dx_norm - np.sum(dx_norm, axis=0) - x_norm * np.sum(dx_norm * x_norm, axis=0))
    
    return dx, dgamma, dbeta

def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization."""
    eps = ln_param.get("eps", 1e-5)
    N, D = x.shape
    
    # Compute mean and variance along the feature dimension (axis=1)
    sample_mean = np.mean(x, axis=1, keepdims=True)
    sample_var = np.var(x, axis=1, keepdims=True)
    
    x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_norm + beta
    
    cache = (x, x_norm, sample_mean, sample_var, gamma, beta, eps)
    
    return out, cache

def layernorm_backward(dout, cache):
    """Backward pass for layer normalization."""
    x, x_norm, mu, var, gamma, beta, eps = cache
    N, D = x.shape
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    
    dx_norm = dout * gamma
    
    dx = (1. / D) * (1. / np.sqrt(var + eps)) * (D * dx_norm - np.sum(dx_norm, axis=1, keepdims=True) - x_norm * np.sum(dx_norm * x_norm, axis=1, keepdims=True))
    
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """Forward pass for dropout."""
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache

def dropout_backward(dout, cache):
    """Backward pass for dropout."""
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """Forward pass for convolution."""
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    out = np.zeros((N, F, H_out, W_out))
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    x_slice = x_pad[n, :, h_start:h_end, w_start:w_end]
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]
    
    cache = (x, w, b, conv_param, x_pad)
    return out, cache

def conv_backward_naive(dout, cache):
    """Backward pass for convolution."""
    x, w, b, conv_param, x_pad = cache
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))
    
    dx_pad = np.zeros_like(x_pad)
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    dw[f] += x_pad[n, :, h_start:h_end, w_start:w_end] * dout[n, f, i, j]
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]
    
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
    
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """Forward pass for max pooling."""
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    
    N, C, H, W = x.shape
    
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    
    out = np.zeros((N, C, H_out, W_out))
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width
                    out[n, c, i, j] = np.max(x[n, c, h_start:h_end, w_start:w_end])
    
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """Backward pass for max pooling."""
    x, pool_param = cache
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape
    
    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width
                    
                    window = x[n, c, h_start:h_end, w_start:w_end]
                    max_val = np.max(window)
                    mask = (window == max_val)
                    dx[n, c, h_start:h_end, w_start:w_end] += dout[n, c, i, j] * mask
    
    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for spatial batch normalization."""
    N, C, H, W = x.shape
    
    # Reshape to (N*H*W, C)
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    
    # Apply batch normalization
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    # Reshape back to (N, C, H, W)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    
    return out, cache

def spatial_batchnorm_backward(dout, cache):
    """Backward pass for spatial batch normalization."""
    N, C, H, W = dout.shape
    
    # Reshape to (N*H*W, C)
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    
    # Apply batch normalization backward
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    
    # Reshape back to (N, C, H, W)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    
    return dx, dgamma, dbeta

def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Forward pass for spatial group normalization."""
    eps = gn_param.get("eps", 1e-5)
    N, C, H, W = x.shape
    
    # Reshape to (N, G, C//G, H, W)
    x_reshaped = x.reshape(N, G, -1, H, W)
    
    # Compute mean and variance per group
    mean = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
    
    # Normalize
    x_norm = (x_reshaped - mean) / np.sqrt(var + eps)
    
    # Reshape back
    x_norm = x_norm.reshape(N, C, H, W)
    
    # Scale and shift
    out = gamma * x_norm + beta
    
    cache = (x, x_norm, mean, var, gamma, beta, eps, G)
    
    return out, cache

def spatial_groupnorm_backward(dout, cache):
    """Backward pass for spatial group normalization."""
    x, x_norm, mean, var, gamma, beta, eps, G = cache
    N, C, H, W = x.shape
    
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
    
    dx_norm = dout * gamma
    
    # Reshape for group normalization
    dx_norm_reshaped = dx_norm.reshape(N, G, -1, H, W)
    x_reshaped = x.reshape(N, G, -1, H, W)
    x_norm_reshaped = x_norm.reshape(N, G, -1, H, W)
    
    # Compute gradients
    N_group = x_reshaped.shape[2] * H * W
    dmean = -np.sum(dx_norm_reshaped, axis=(2, 3, 4), keepdims=True) / np.sqrt(var + eps)
    dvar = -0.5 * np.sum(dx_norm_reshaped * (x_reshaped - mean), axis=(2, 3, 4), keepdims=True) * (var + eps)**(-1.5)
    
    dx = (dx_norm_reshaped / np.sqrt(var + eps) + 
          dmean / N_group + 
          2 * dvar * (x_reshaped - mean) / N_group)
    
    dx = dx.reshape(N, C, H, W)
    
    return dx, dgamma, dbeta