import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D





def line_diff(p,v,q,w):
    
    q_ = q - p

    n0 = v/np.linalg.norm(v)

    n1 = w - np.dot(w,n0)*n0

    n1 = n1/np.linalg.norm(n1)

    diff = q_ - np.dot(q_,n0)*n0 - np.dot(q_,n1)*n1
    
    t0 = - np.dot(q_, n1)/np.dot(w, n1)
    
    return t0*w + q_ + p , -diff 
    


fig = plt.figure()
ax = fig.gca(projection='3d')


X  = np.random.sample((2,3))


D = np.random.randn(2,3)
D = D/np.linalg.norm(D, axis=1, keepdims=True)
D = D*np.random.sample((2,1))*2.


ax.plot([X[0,0] - D[0,0], X[0,0] + D[0,0]], 
	    [X[0,1] - D[0,1], X[0,1] + D[0,1]],
	    [X[0,2] - D[0,2], X[0,2] + D[0,2]], label='1')

ax.plot([X[1,0]- D[1,0], X[1,0] + D[1,0]], 
	    [X[1,1]- D[1,1], X[1,1] + D[1,1]],
	    [X[1,2]- D[1,2], X[1,2] + D[1,2]], label='2')



base, diff = line_diff(X[0], D[0], X[1], D[1])

# no = np.array(no)
# ax.scatter(no[0,0],no[0,1], no[0,2])
# ax.scatter(no[1,0],no[1,1], no[1,2])
# ax.scatter(no[2,0],no[2,1], no[2,2])
ax.scatter(base[0],base[1],base[2])



ax.plot([base[0], base[0] + diff[0]], 
	    [base[1], base[1] + diff[1]],
	    [base[2], base[2] + diff[2]], label='diff')

ax.legend()
plt.show()