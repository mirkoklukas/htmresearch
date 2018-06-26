import numpy as np

def create_basis(theta, phi=np.pi/2.):
    return np.array([
        [np.cos(theta), np.cos(theta + phi)],
        [np.sin(theta), np.sin(theta + phi)]
    ])


def create_2d_hyper_module(m, scale=None):

    k=2
    V = np.zeros((m,k, 2))
    B = np.zeros((m, k, k))
    S = np.zeros(m)
    M = {}
    for i in range(m):
        if scale is not None:
            S[i] = scale[i]
        else:
            S[i] = np.sqrt(2)**i

    for i in range(m):
        theta = np.random.sample()*2.*np.pi
        R = create_basis(theta)
        B[i, :, :] = S[i]*R

    return (B, V, S)


def create_action_tensor(M):
    B, _, _ = M
    m, k, _ = B.shape
    A  = np.zeros((m, 2, k))
    P  = np.zeros((2, k))
    P[0,0] = 1.
    P[1,1] = 1. 

    for i in range(m):
        A[i] = np.dot(P, np.linalg.inv(B[i]))

    return A





def pipe_through_tensor(A, V):

    m, _, _ = A.shape
    T, _ = V.shape
    V_ = np.zeros((T, m,2))
    for i in range(m):
        V_[:, i, :] = np.dot(V, A[i].T)

    return V_


def apply_phasechange(p, v):
    return (p + v)%1




def create_atlas(nr, nc_, diam=1.0):
    nc = nc_**2
    A = np.zeros((nr, nc, 2))
    for i in range(nr):
        box = np.indices((nc_,nc_)).reshape((2,-1)).T
        box = box/float(nc_)
        A[i] = box

    return A*diam






def create_phase_atlas(PC, Anchors, A):
    nr, nc, _ = PC.shape
    V = PC - PC[:,[0],:] 
    m = len(A)
    
    Phi = np.zeros((nr, nc, m, 2))
    for r in range(nr):
            V_ =  pipe_through_tensor(A, V[r])
            Phi[r] = apply_phasechange(Anchors[[r]], V_)

    return Phi


def smoothstep(x):
    x = np.clip(x, 0.,1.)
    y = np.where( (x > 0) & (x<1), 3*x**2 - 2*x**3, x)
    return y





def phase_dist(P, Q):
    D = np.minimum( np.absolute(P - Q), 1. - np.absolute(P-Q))
    D = np.linalg.norm(D, axis=2)
    # D = smoothstep(2*D)
    PD = np.linalg.norm(D, axis=1)
    return PD







def module_dist(P, Q, M):
    B, _, S = M
    T,  _ = P.shape
    dist  = np.zeros(T)
    offsets = np.array([(0.,0.), (1.,0.), (1.,1.),(0.,1.),(-1.,1.),(-1.,0.),(-1.,-1.),(0.,-1.), (1.,-1.)])
    for t in range(T):
            p = P[t]
            q = Q[t]
            p_ = np.dot(p, B.T)
            Q_ = np.dot(q + offsets, B.T)
            D = Q_ - p_

            L = np.linalg.norm(D, axis=1)
            dist[t] = np.amin(L)
            # l = np.argmin(L)

    return dist




