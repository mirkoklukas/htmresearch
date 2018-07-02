import numpy as np



def relu(x):
    return np.maximum(x, 0.)


def evolve_step(W, b, s, beta=0., mask=1.):
    n     = W.shape[0]
    dt    = 0.01
    tau   = 1.0
    f     = relu
    noise = np.random.randn(n)*0.001
    
    ds    = dt*(f(np.dot(W,s+ noise) + b + beta) - s/tau)
    s_    = s + ds
    s_ *= mask
    if np.sum(s_**2) >0:
            s_ = s_/np.sqrt(np.sum(s_**2))
            
    return s_

