import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np

from sklearn.preprocessing import StandardScaler

def scale(data, predictors):
    scaler = StandardScaler()
    data[predictors] = scaler.fit_transform(data[predictors])
    return data

def init_layers(inputs):
    layers = []
    for i in range(1, len(inputs)):
        layers.append([
            np.random.rand(inputs[i-1], inputs[i]) / 5 - .1,
            np.ones((1,inputs[i]))
        ])
    return layers

def forward(layers, x):
    for i in range(len(layers)):
        x = jnp.matmul(x, layers[i][0]) + layers[i][1]
        if i < len(layers) - 1:
            x = jnp.maximum(x, 0)
    return x

def mse(y, preds):
    return jnp.mean((y - preds)**2)

def loss(layers, x, y):
    preds = forward(layers, x) 
    return mse(y, preds)

@jit
def backward(layers, x, y, lr):
    grads = grad(loss)(layers, x, y)
    for layer, g in zip(layers, grads):
        layer[0] -= (g[0] + layer[0] * .01) * lr 
        layer[1] -= g[1] * lr
    return layers