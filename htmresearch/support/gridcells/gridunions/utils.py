import numpy as np




def create_orthogonal_basis(theta):
    return np.array([
        [np.cos(theta), np.cos(theta + np.pi/2.)],
        [np.sin(theta), np.sin(theta + np.pi/2.)]
    ])


def create_action_tensor(M):
    B, _, _ = M
    m, k, _ = B.shape
    A  = np.zeros((m, 2, k))
    Pr = np.zeros((2, k))
    Pr[0,0] = 1.
    Pr[1,1] = 1. 

    for i in range(m):
        A[i] = np.dot(Pr, np.linalg.inv(B[i]))

    return A


def create_2d_hyper_module(m, scale=None):

    k=2
    V = np.zeros((m,k, 2))
    B = np.zeros((m, k, k))
    S = np.zeros(m)
    
    for i in range(m):
        if scale is not None:
            S[i] = scale[i]
        else:
            S[i] = np.sqrt(2)**i

    for i in range(m):
        theta = np.random.sample()*2*np.pi
        B[i, :, :] = S[i]*create_orthogonal_basis(theta)


    return (B, V, S)



def create_atlas(nr, nc_, diam=1.0):
    nc = nc_**2
    A = np.zeros((nr, nc, 2))
    for i in range(nr):
        box = np.indices((nc_,nc_)).reshape((2,-1)).T
        box = box/float(nc_)
        A[i] = box

    return A*diam



def apply_phasechange(phi, v):
    phi_  = phi + v
    phi_ %= 1
    return phi_


def phase_dist(P, Q):
    D = np.minimum( np.absolute(P - Q), 1. - np.absolute(P-Q))
    D = np.linalg.norm(D, axis=2)
    PD = np.linalg.norm(D, axis=1)
    return PD


def pipe_through_tensor(A, V):

    m, _, _ = A.shape
    T, _ = V.shape
    V_ = np.zeros((T, m,2))
    for i in range(m):
        V_[:, i, :] = np.dot(V, A[i].T)

    return V_

def create_phase_atlas(PC, Anchors, A):
    nr, nc, _ = PC.shape
    V = PC - PC[:,[0],:] 
    m = len(A)
    
    Phi = np.zeros((nr, nc, m, 2))
    for r in range(nr):
            V_ =  pipe_through_tensor(A, V[r])
            Phi[r] = apply_phasechange(Anchors[[r]], V_)

    return Phi




