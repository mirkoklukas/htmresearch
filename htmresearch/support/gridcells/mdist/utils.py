import numpy as np


def create_orthogonal_basis(theta=0.):
    return np.array([
        [np.cos(theta), np.cos(theta + np.pi/2.)],
        [np.sin(theta), np.sin(theta + np.pi/2.)]
    ])


def create_hyper_module(m, S):
    V = np.random.multivariate_normal(mean=np.zeros(2), cov=100.*np.eye(2), size=m)
    B = np.zeros((m,2,2))
    Theta = np.random.sample(m)*2*np.pi
    # Theta = np.random.normal(0.0,0.1,size=m)
    # Theta = np.zeros(m)

    for i in range(m):
        B[i] = S[i]*create_orthogonal_basis(Theta[i])
    
    return B, V, S, Theta


def create_3d_hyper_module(m, scale=None):


    V = np.random.multivariate_normal(mean=np.zeros(3), cov=100.*np.eye(3), size=m)
    B = np.zeros((m, 3, 3))
    S = np.zeros(m)
    
    for i in range(m):
        if scale is not None:
            S[i] = scale[i]
        else:
            S[i] = np.sqrt(2)**i

    for i in range(m):
        theta = np.random.sample()*2*np.pi
        B[i, :2, :2] = S[i]*create_orthogonal_basis(theta)

        theta = np.random.sample()*2*np.pi
        B[i, :2, 2]  = S[i]*np.array([np.cos(theta), np.sin(theta)])
        B[i,  2, 2]  = S[i]
        # b3  = np.random.randn(3)
        # b3 /= np.linalg.norm(b3)
        # B[i,:,2] = S[i]*b3
        
    return B, V, S


def create_kd_hyper_module(m, k, scale=None):


    V = np.random.multivariate_normal(mean=np.zeros(k), cov=100.*np.eye(k), size=m)
    B = np.zeros((m, k, k))
    S = np.zeros(m)
    
    for i in range(m):
        if scale is not None:
            S[i] = scale[i]
        else:
            S[i] = np.sqrt(2)**i

    for i in range(m):
        theta = np.random.sample()*2*np.pi
        B[i, :2, :2] = S[i]*create_orthogonal_basis(theta)

        # theta = np.random.sample()*2*np.pi
        # B[i, :2, 2]  = S[i]*np.array([np.cos(theta), np.sin(theta)])
        # B[i,  2, 2]  = S[i]
        for l in range(2,k):
            b  = np.random.randn(k)
            b /= np.linalg.norm(b)
            B[i,:,l] = S[i]*b

        
    return B, V, S


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


def map_to_quotient(X, B, v=0.0):
    X_ = X - v
    Y = np.dot(X_, np.linalg.inv(B).T)
    Y %= 1
    return Y[:,:2]



def map_to_hypertorus(B, V, X):
    m = len(B)
    T = len(X)
    Y = np.zeros((T, m, 2))
    
    for i in range(m):
        Y[:,i,:] = map_to_quotient(X, B[i], V[i])
        
    return Y

def map_3d_to_quotient(X, B, v):
    X_ = X - v
    Y = np.dot(X_, np.linalg.inv(B).T)
    Y %= 1
    return Y[:,:2]


def map_3d_to_hypertorus(B, V, X):
    m = len(B)
    T = len(X)
    Y = np.zeros((T, m, 2))
    
    for i in range(m):
        Y[:,i,:] = map_3d_to_quotient(X, B[i], V[i])
        
    return Y


def M_dist_comp(P, Q, S=None):

    D = np.minimum( np.absolute(P - Q), 1. - np.absolute(P-Q))

    if S is not None:
        D *= S.reshape((1,-1,1))

    D = np.linalg.norm(D, axis=2)

    return D


def M_dist(P, Q, S=None):
    mdc = M_dist_comp(P, Q, S)
    md = np.linalg.norm(mdc, axis=1)

    return md 


def M_dist_comp_unskewed(P, Q, B, S=None):

    T,  _ = P.shape
    dist = np.zeros(T)
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










