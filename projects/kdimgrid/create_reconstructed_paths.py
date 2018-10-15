import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append("../")
from unique_hypercube_size import *

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm




def random_run(T,k):
	X = np.zeros((T,k))
	v = np.random.randn(k)
	v = v/np.sqrt(np.sum(v**2.))*0.008
	a = np.zeros(k)

	# X[0] = np.ones(k)*0.5 
	# X[0] = 0.5 + (np.random.sample(k)-0.5)*0.2/0.5
	X[0] = np.random.sample(k)
	# X[0, 0] = 0.

	# v[0] = .07
	# v[1:] = 0.



	for t in range(1,T):
		v += a
		v = np.clip(v, -0.008,0.008)


		X[t] = X[t-1] + v
		a = np.random.randn(k)*0.002

		for i in range(0,k):
			if X[t, i] >= 1.:
				v[i] *= -1.
				a[i] = 0.
			if X[t, i] < 0.:
				v[i] *= -1.
				a[i] = 0.


		a = np.clip(a, -0.002,0.002)

	np.clip(X, 0., 1., X)
	return X


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax = fig.gca(projection='3d')

T=300
t = np.linspace(0.,2*np.pi, num=T)

# g = np.zeros((T,3))
# g[:,0] = np.sin(t) + 2.*np.sin(2.*t)
# g[:,1] = np.cos(t) - 2.*np.cos(2.*t)
# g[:,2] = - np.sin(3.*t) 

# g = g/6. + 0.5

# ax.plot(g[:,0], g[:,1], g[:,2] )
# ax.scatter(g[:,0], g[:,1], g[:,2] )
# plt.show()



k = 3


n=50**3
"""
First sample is the zero vector
"""


"""
First uniformly sample 
from $k$-dim ball of radius r
"""


def phase(X, A_):
    return np.dot(A_,X.T).T%1.

def reconstruct(phi, X, Y):

    phi = phi.reshape((1,-1))

    diff = (Y - phi)%1.


    diff = np.amax(np.minimum( diff, 1. - diff), axis=1)
    i = np.argmin(diff)
    return X[i]
 




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Get rid of colored axes planes

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# # Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')


bins =50

scale = 5.

ax.set_xlim(0.,scale)
ax.set_ylim(0.,scale)
ax.set_zlim(0.,scale)
max_m = 2


m=6
# for m in range(2,max_m):
	# print m
S = np.sqrt(2)**np.arange(m)
print S
d = []




# U = np.random.uniform(size=(n,k))*scale
U = np.indices((bins,bins,bins)).T
U = U.reshape((-1,3)).astype(float)
U = ((U+0.5)/float(bins))*scale
print U.shape



for j in range(1):
	X  = random_run(T,3)*scale
	# # X = g*scale
	# ax.scatter(X[0,0],X[0,1],X[0,2],zorder=100, marker="o", s=40,facecolor="white", color=plt.cm.winter(0.))
	# ax.scatter(X[-1,0],X[-1,1],X[-1,2],zorder=100, marker="o", s=40, facecolor="white", color=plt.cm.winter(1.))
	print j, "done"
	print "X=", repr(X)
	for i in range(1):


		A = create_random_A(m, k, S)
		A_ = A.reshape((2*m,k))

		g_ = np.zeros((T,k))

		C = phase(U, A_) 

		# vox = np.zeros((bins,bins,bins))

		X_ = np.zeros((T,3))
		for t in range(T):
		    X_[t] = reconstruct(phase(X[t], A_), U, C)
		    # a,b,c = (X_[t]/scale*bins).astype(int).tolist()
		    # vox[a,b,c] = 1

		print "X_=",repr(X_)


		    
		print "\t", i, "	done"



		# X_ = X_/scale



		

		d = d + np.sqrt(np.sum( (X_-X)**2., axis=1)).tolist()
		d = d-np.min(d)
		d = d/np.max(d)
		# ax.plot(X[:,0], X[:,1], X[:,2], alpha=.5, linewidth=1., color="darkgray")
		# for t in range(T):
			# ax.plot(X[t:t+2,0], X[t:t+2,1], X[t:t+2,2], linewidth=1., zorder=X[t,2], color="C{}".format(i))



		# c = X_[:,1] 
		# c = c- np.amin(c)
		# c= c/np.amax(c)
		# c = c
		c=np.linspace(0.,1,num=T)
		# ax.plot(X_[:,0], X_[:,1], X_[:,2], linewidth=1., alpha=.4, zorder=-1, color="C{}".format(j))#color="darkgray")


		# ax.voxels(vox, edgecolor=(1.,0,0,1.), linewidth=0.01, facecolor=(1.,0,0,0.1))
		for t in range(T):
			# ax.plot([X[t,0],X_[t,0]],[X[t,1],X_[t,1]], [X[t,2],X_[t,2]], linewidth=1., zorder=-X_[t,1] + 0.02*scale, alpha=1.,  color=plt.cm.winter(c[t]))#color="darkgray")
			# ax.plot(X_[t:t+2,0], X_[t:t+2,1], X_[t:t+2,2], linewidth=.5, zorder=-X[t,1]+ 0.02*scale, alpha=1., color="blue")
			# ax.plot(X_[t:t+2,0], X_[t:t+2,1], X_[t:t+2,2], linewidth=1., zorder=-X[t,1]+ 0.02*scale, color=plt.cm.winter(c[t]))
			ax.plot(X_[t:t+2,0], X_[t:t+2,1], X_[t:t+2,2], linewidth=1., zorder=-X_[t,1]+ 0.02*scale, color=plt.cm.winter(c[t]))

	# ax.plot(X[:,0], X[:,1], X[:,2], linewidth=1., alpha=1., color="C{}".format(j))
	# c = X[:,1] 
	# c = c- np.amin(c)
	# c= c/np.amax(c)
	# c = c
	c=np.linspace(0.,1,num=T)

	# X = X/scale*bins
	for t in range(T):
		ax.plot(X[t:t+2,0], X[t:t+2,1], X[t:t+2,2], linewidth=2., zorder=-X[t,1]+ 0.02*scale, color=plt.cm.winter(c[t]))

	for t in range(T):
		ax.plot(X[t:t+2,0], X[t:t+2,1], X[t:t+2,2], linewidth=6., zorder=-X[t,1] - 0.02*scale, color="white")



# for t in range(T):
	# ax.plot(g[t:t+2,0], g[t:t+2,1], g[t:t+2,2],  color="white", linewidth=13., zorder=g[t,2]-0.1)
# for t in range(T):
	# ax.plot(g[t:t+2,0], g[t:t+2,1], g[t:t+2,2],  color="black", linewidth=1., zorder=g[t,2]+0.05, linestyle="-")




# ax.plot(g[:,0], g[:,1], g[:,2], alpha=1, linewidth=1., color='blue', linestyle="-")
# ax.scatter(g_[:,0], g_[:,1], g_[:,2])
# ax.scatter(g[:,0], g[:,1], g[:,2] )
ax.grid(False)

ax.view_init(elev=6., azim=-81)
plt.savefig("./paths/reconstruct_{}.pdf".format(np.random.randint(100000)))
plt.show()


# plt.hist(d)
# plt.savefig("./histogram.pdf")
# plt.show()




# D   = np.random.multivariate_normal(mean=np.zeros(k),cov=np.eye(k), size=n) 
# D   = D/ np.sqrt(np.sum(D**2, axis=1, keepdims=True))
# U   = np.random.uniform(size=(n,1))
# X_r = r*D*(U**(1./k))
# X_ = np.concatenate([X_, X_r], axis=0)

