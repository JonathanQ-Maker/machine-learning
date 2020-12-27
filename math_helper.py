import numpy as np

def sigmoid(x):
    '''Accepts non numpy array'''
    return 1 / (1 + np.exp(-x))

def relu(x):
    '''Accepts non numpy array'''
    return np.maximum(x, 0)

def probability_dist(x):
    '''Accepts non numpy array'''
    return x / np.sum(x)

def softmax(x, axis=None):
    '''Accepts non numpy array'''
    r = np.exp(x)
    if(axis ==1):
        return r / np.reshape(np.sum(r, axis=1), (-1, 1))
    return r / np.sum(r)

def duel_softmax(x, axis=None):
    '''Accepts non numpy array'''
    r = np.exp(x)
    if(axis ==1):
        s = np.reshape(np.sum(r, axis=1), (-1, 1))
    else:
        s = np.sum(r)
    return r / s, s

def stable_duel_softmax(x, axix=None):
    """Compute the softmax of vector x in a numerically stable way."""
    exps = np.exp(x - np.max(x))
    if (axis == 1):
        s = np.reshape(np.sum(exps, axis=1), (-1, 1))
    else:
        s = np.sum(exp)
    return exps / s, s

def normalize(arry):
    '''Accepts non numpy array'''
    max = np.amax(np.abs(arry))
    if max == 0.0:
        return arry
    return arry / max

def reshape_2d21d(arry):
    r = []
    for y in arry:
        for x in y:
            r.append(x)
    return r

def softmax_gradient(z_output, output, softmax_sum):
    return (output / softmax_sum) * (np.exp(2 * z_output) / softmax_sum + 1)