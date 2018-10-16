import numpy as np 



def create_block_A(m, k, S):


    A = np.zeros((m,2,k))
    
    for i_ in range(k): 
        for j in range(m//k):
            s = S[j]
            i = i_*(m//k) + j
            a  = np.random.randn(2)
            a /= np.linalg.norm(a)

            A[i,:,i_] = a / s


    return A

def create_random_A_normal(m, k):

    A = np.random.standard_normal(size=(m,2,k))

    return A

def create_random_A_shuffled(m, k, S):

    A = np.zeros((m*2,k))
    for l in range(k):
      v = np.zeros((m,2))
      for i in range(m):
        a  = np.random.randn(2)
        a /= np.linalg.norm(a)
        v[i,:] = a 
      S_ = S[np.random.permutation(m)].reshape((m,1))

      A[:,l] = (v/S_).reshape(2*m)

    return A.reshape((m,2,k))


def create_random_A(m, k, S):

    A = np.zeros((m,2,k))
    for i, s in enumerate(S):
        for l in xrange(k):
            a  = np.random.randn(2)
            a /= np.linalg.norm(a)
            A[i,:,l] = a / s

    return A
