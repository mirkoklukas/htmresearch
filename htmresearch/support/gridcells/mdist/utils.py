import numpy as np
from scipy.special import gamma as Gamma


def ball_volume(k, r=1.):
    return np.pi**(k/2.) / Gamma(k/2. + 1) * (r**k)


def create_orthogonal_basis(theta=0.):
    return np.array([
        [np.cos(theta), np.cos(theta + np.pi/2.)],
        [np.sin(theta), np.sin(theta + np.pi/2.)]
    ])


def create_random_A(m,k,S):

    A = np.zeros((m, 2, k))
    for i, s in enumerate(S):
        for l in xrange(k):
            a  = np.random.randn(2)
            a /= np.linalg.norm(a)
            A[i,:,l] = a / s

    return A

 


def create_module_basis(m, k, S):

    B = np.zeros((m, k, k))

    for i in range(m):
        theta = np.random.sample()*2*np.pi
        if k>1:
            B[i, :2, :2] = S[i]*create_orthogonal_basis(theta)
        else:
            B[i, 0, 0] = S[i]*np.random.choice([-1.,1.])

        for l in range(2,k):
            b  = np.random.randn(2)
            b /= np.sqrt(np.sum(b**2))
            
            B[i,:2,l] = S[i]*b[:]

            B[i,l,l] = S[i]
            # B[i,l,l] = 1.


        
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


def map_to_quotient_A(X, A):
    X_ = X 
    Y = np.dot(X_, A.T)
    Y %= 1
    return Y



def map_to_hypertorus(B, X):
    m = len(B)
    T = len(X)
    Y = np.zeros((T, m, 2))
    
    for i in range(m):
        Y[:,i,:] = map_to_quotient(X, B[i])
        
    return Y

def map_to_hypertorus_A(A, X):
    m = len(A)
    T = len(X)
    Y = np.zeros((T, m, 2))
    
    for i in range(m):
        Y[:,i,:] = map_to_quotient_A(X, A[i])
        
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


def M_dist_2(P, Q, S=None):

    D = np.minimum( np.absolute(P - Q), 1. - np.absolute(P-Q))

    if S is not None:
        D *= S.reshape((1,-1,1))

    D = D.reshape((-1, D.shape[1]*D.shape[2]))
    D = np.amax(D, axis=1)

    return D






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