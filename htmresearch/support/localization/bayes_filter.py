import numpy as np 


def bayes_filter_update(b, T_u, m):
	pred = np.dot(T_u, b)
	b_   = m*pred.reshape(-1)
	b_   = b_/np.sum(b_) 
	return b_





