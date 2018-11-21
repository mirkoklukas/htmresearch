import numpy as np



def relu(x):
    return np.maximum(x, 0.)


def evolve_time_step(W, b, s, dt=0.01, tau=2., f=relu):
    n     = W.shape[0]
    # noise = np.random.randn(n)*0.001
    # noise = np.zeros(n)
    Ws = np.dot(W,s)
    ds    = ( f(Ws + b ) - s/tau )*dt
    s_    = s + ds

            
    return s_


def evolve_step(W, b, s, beta=0., mask=1.):
    n     = W.shape[0]
    dt    = 0.01
    tau   = 2.0
    f     = relu
    # noise = np.random.randn(n)*0.001
    noise = np.zeros(n)
    
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


# def evolve(W, b, s, A=1., dt=0.01, tau=.03, f=relu):
def evolve(W, b, s, A=1., dt=0.05, tau=3., f=relu):

    I  = A*(np.dot(W,s) + b )

    
    ds = dt*(f(I) - s/tau)
    s_ = s + ds
            
    return s_




def run_can(T, W, B, S, dt=0.05, tau=3., f=relu, ):



    for t in range(1,T):
        s = S[t-1]
        b = B[t-1]
        # b += eventhandler[events[t-1]](s)
        
        Ws = np.dot(W,s)
        ds  = ( f(Ws + b ) - s/tau )*dt
        s_  = s + ds

        S[t] = s_



