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