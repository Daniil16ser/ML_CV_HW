import numpy as np

def affine_forward(x, w, b):
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    out = x_flat @ w + b
    cache = (x, w, b, x.shape)
    return out, cache

def affine_backward(dout, cache):
    x, w, b, x_shape = cache
    
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    
    dw = x_flat.T @ dout
    dx_flat = dout @ w.T
    dx = dx_flat.reshape(x_shape)
    db = np.sum(dout, axis=0)
    
    return dx, dw, db

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx

def affine_relu_forward(x, w, b):
    out_affine, cache_affine = affine_forward(x,w,b)
    
    out_relu, cache_relu = relu_forward(out_affine)

    cache = (cache_affine, cache_relu)
    return out_relu, cache


def affine_relu_backword(dout, cache):
    cache_affine, cache_relu = cache

    dout_affine = relu_backward(dout, cache_relu)

    dx, dw, db = affine_backward(dout_affine, cache_affine)

    return dx, dw, db


# Простая проверка relu_backward
print("Проверка relu_backward:")

# Создаем простые данные
x_test = np.array([-2, -1, 0, 1, 2])
dout_test = np.array([1, 1, 1, 1, 1])

# Forward
out_test, cache_test = relu_forward(x_test)
print("out:", out_test)  # должно быть [0, 0, 0, 1, 2]

# Backward
dx_test = relu_backward(dout_test, cache_test)
print("dx:", dx_test)  # должно быть [0, 0, 0, 1, 1]
print("dx is None?", dx_test is None)