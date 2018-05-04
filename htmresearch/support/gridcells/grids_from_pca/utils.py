import numpy as np
from scipy.ndimage.interpolation import shift




def oja(W, X, learning_rate=0.1, non_negative=False):


	ONES = np.ones((W.shape[0],W.shape[0]))


	L = - np.tril(ONES, k=1)
	np.fill_diagonal(L, 1.0)
	for t in range(len(X)):

		x  = X[[t]].T
		y  = np.dot(W,x) 

		dW = np.zeros(W.shape)
		for i in range(W.shape[0]):
			for j in range(W.shape[1]):
				c = np.sum( W[k,j]*y[k,0] for k in range(i+1)) 
				dW[i,j] = y[i,0]*x[j,0] - y[i,0]*c 


		W[:,:] = W + learning_rate*dW

		if non_negative:
			W[:,:] = np.maximum(W,0.0)[:,:]




def oja_w1(W, X, learning_rate=0.1, non_negative=False):


	for t in range(len(X)):

		x  = X[[t]].T
		y  = np.dot(W,x) 


		dW = y*x.T - y*y*W 


		W[:,:] = W + learning_rate*dW

		if non_negative:
			W[:,:] = np.maximum(W,0.0)[:,:]





def transition_from_matrix(T):
	n = T.shape[1]
	return lambda i: np.random.choice(n, p=T[i])


def gaussian_encoder(env_shape, wrap=False, sigma = 10.):
	h, w = env_shape
	model_bump = np.zeros(env_shape)
	mu0 = w//2
	mu1 = h//2

	for i in range(h):
		for j in range(w):
			dist = np.sqrt((i-mu0)**2 + (j-mu1)**2)
			y = np.exp( - dist**2/sigma)
			model_bump[i,j] = y


	def encode(n):
		i = n//h
		j = n%h
		if wrap:
			bump = np.roll(model_bump, i - mu0, axis=0)
			bump = np.roll(bump, j - mu1, axis=1)
		else:
			bump = shift(model_bump, (i - mu0,j - mu1))

		return bump.reshape(-1)

	return encode


def create_transition_matrix(n_, wrap=True, sigma= 2., with_diagonal=False):
	n = n_**2
	T = np.zeros((n,n))

	enc = gaussian_encoder((n_,n_), wrap, sigma)

	for t in range(n):
		T[t] = enc(t)


	for t in range(n):
		if with_diagonal == False:
			T[t,t] = 0.0

		T[t] /= np.sum(T[t]) 

	return T


def create_random_walk_from_transition_fct(transition_fct, encoding_fct, n, num_data_points=1000):
	X = np.zeros((num_data_points,n))
	T = transition_fct
	E = encoding_fct

	i = np.random.choice(n)
	for t in range(num_data_points):
		X[t] = E(i)
		i    = T(i)

	return X

