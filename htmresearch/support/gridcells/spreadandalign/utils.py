import numpy as np



def get_1d_loop(exploration_range=10, speed=1.):
	r = exploration_range
	X = []
	V = []
	x = 0
	X.append(0)
	while x < r:
		dx = np.random.rand()
		x += dx
		V.append(dx)
		X.append(x)
	while x>0:
		dx = np.random.rand()
		x -= dx
		V.append(-dx)
		X.append(x)

	X[-1] = 0
	V[-1] = - (x - V[-1])

	return np.array(X).reshape((1,-1)), np.array(V).reshape((1,-1))




