import numpy as np


def create_orthogonal_basis(theta=0.):
    return np.array([
        [np.cos(theta), np.cos(theta + np.pi/2.)],
        [np.sin(theta), np.sin(theta + np.pi/2.)]
    ])




def create_module_basis(m, k, S):
    B = np.zeros((m, k, k))

    for i in range(m):
        theta = np.random.sample()*2*np.pi
        B[i, :2, :2] = S[i]*create_orthogonal_basis(theta)

        for l in range(2,k):
            b  = np.random.randn(k)
            b /= np.linalg.norm(b)
            B[i,:,l] = b

        
    return B


def create_action_tensor(B):
    m, k, _ = B.shape
    A  = np.zeros((m, 2, k))
    Pr = np.zeros((2, k))
    Pr[0,0] = 1.
    Pr[1,1] = 1. 

    for i in range(m):
        A[i] = np.dot(Pr, np.linalg.inv(B[i]))

    return A


def pipe_through_tensor(A, V):
    m, _, k = A.shape
    T, _ = V.shape
    V_ = np.zeros((T, m,2))
    for i in range(m):
        V_[:, i, :] = np.dot(V, A[i].T)

    return V_


def apply_velocity(P, V):
    P_  = P + V
    P_ %= 1

    return P_


def map_to_quotient(X, B):
    X_ = X 
    Y = np.dot(X_, np.linalg.inv(B).T)
    Y %= 1
    return Y[:,:2]



def map_to_hypertorus(B, X):
    m = len(B)
    T = len(X)
    Y = np.zeros((T, m, 2))
    
    for i in range(m):
        Y[:,i,:] = map_to_quotient(X, B[i])
        
    return Y


def M_dist_comp(P, Q, S=None):

    D = np.minimum( np.absolute(P - Q), 1. - np.absolute(P-Q))

    if S is not None:
        D *= S.reshape((1,-1,1))

    D = np.linalg.norm(D, axis=2)

    return D


def M_dist(P, Q, S=None):
    mdc = M_dist_comp(P, Q, S)
    md  = np.linalg.norm(mdc, axis=1)

    return md 



def M_dist_max(P, Q, S=None):
    mdc = M_dist_comp(P, Q, S)
    md  = np.amax(mdc, axis=1)

    return md 




def geod_on_hypertorus(P,Q):
    T, m = P.shape[:2]
    G = np.zeros((T, m, 2))
    for t in range(T):
        for i in range(m):
            G[t,i] = geod(P[t,i], Q[t,i])

    return G


def geod(p,q):
    offsets = np.array([(0.,0.), (1.,0.), (1.,1.),(0.,1.),(-1.,1.),(-1.,0.),(-1.,-1.),(0.,-1.), (1.,-1.)])
    Q = q + offsets
    D = Q - p
    L = np.linalg.norm(D, axis=1)
    l = np.argmin(L)
    return D[l]










