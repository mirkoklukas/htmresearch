import numpy as np
import matplotlib.pyplot as plt


def norm(v):
	return np.sqrt(np.dot(v,v))

def vel(theta):
	return np.array([np.cos(theta), np.sin(theta)])

def adjust(x, theta):
	"""
	If point is on the boundary of the box, add a constant to the angle
	to encourage leaving the boundary.
	"""
	c = .5
	v = vel(theta)
	if x[0] == 0:
		if v[1] > 0:
			theta -= c
		else:
			theta += c

	if x[0] == 1:
		if v[1] > 0:
			theta += c
		else:
			theta -= c

	if x[1] == 0:
		if v[0] > 0:
			theta += c
		else:
			theta -= c

	if x[1] == 1:
		if v[0] > 0:
			theta -= c
		else:
			theta += c

	return x, theta



def smooth_walk(num, start=None, speed=0.01, smoothness=0.5):

	if start == None:
		x = np.random.randn(2)
	else:
		x = np.array(start)

	X     = np.zeros((2,num))
	theta = np.random.randn()

	for t in range(num):
		theta += np.random.normal(0, (1-smoothness)*np.pi)
		x += speed*vel(theta)
		x  = np.clip(x,0,1)

		x, theta = adjust(x, theta)
		X[:,t] = x[:]

	return X

