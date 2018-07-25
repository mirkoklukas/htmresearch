import numpy as np



def relu(x):
    return np.maximum(x, 0.)


def evolve_step(W, b, s, beta=0., mask=1.):
    n     = W.shape[0]
    dt    = 0.01
    tau   = 1.0
    f     = relu
    noise = np.random.randn(n)*0.001
    # noise = np.zeros(n)
    
    ds    = dt*(f(np.dot(W,s+ noise) + b + beta) - s/tau)
    s_    = s + ds
    s_ *= mask
    if np.sum(s_**2) >0:
            s_ = s_/np.sqrt(np.sum(s_**2))
            
    return s_



def create_envelope(n, steepness, delta):
    x  = np.linspace(0.,1.,num=n)
    x_ = np.absolute(x - 0.5)/delta - 0.5/delta + 1
    x_ [x_< 0] = 0
    # A = np.exp(-1.*steepness * ((x_ - 1. + delta)/delta)**2)
    A = np.exp(- steepness *x_**2)

    return A


def evolve(W, b, s, A=1., dt=0.05, tau=3., f=relu):

    I  = A*(np.dot(W,s) + b )
    ds = dt*(f(I) - s/tau)
    s_ = s + ds

    if np.sum(s_**2) >0:
        s_ = s_/np.sqrt(np.sum(s_**2))
            
    return s_