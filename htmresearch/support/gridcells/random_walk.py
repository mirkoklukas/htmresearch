# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


def norm(v):
	return np.sqrt(np.dot(v,v))

def direction(theta):
	return np.array([np.cos(theta), np.sin(theta)])

def adjust(x, theta, speed):
	"""
	If point is on the boundary of the box, add a constant to the angle
	to encourage leaving the boundary.
	"""
	c = .5
	v = direction(theta)
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

	return theta, speed



def smooth_torus_walk(num, start=None, min_speed=0.0, max_speed=0.04, sigma=0.5):

	if start == None:
		x = np.random.randn(2)
	else:
		x = np.array(start)

	X     = np.zeros((num,2))
	X[0]  = x[:]  
	V     = np.zeros((num,2))
	theta = np.random.randn()


	for t in range(1,num):
		speed  = max_speed*np.random.sample()
		dtheta = np.random.normal(loc=0.0, scale=sigma)
		theta += dtheta

		
		vel   = speed*direction(theta)
		x    += vel
		x  %= 1
		X[t,:] = x[:]

		V[t-1,:] = vel	

	return X, V



def is_within_box(x):
	return (x[0] >= 0 and x[0]<= 1 and x[1] >= 0 and x[1]<= 1)

def smooth_walk(num, start=None, min_speed=0.0, max_speed=0.04, sigma=0.5):

	if start == None:
		x = np.random.randn(2)
	else:
		x = np.array(start)

	X     = np.zeros((num,2))
	X[0]  = x[:]  
	V     = np.zeros((num,2))
	theta = np.random.randn()


	for t in range(1,num):
		speed  = min_speed + (max_speed - min_speed)*np.random.sample()
		dtheta = np.random.normal(0, sigma)
		theta  = theta + dtheta
		v      = speed*direction(theta)
		x      = np.clip(x + v, 0, 1)
		theta, speed = adjust(x, theta, speed)

		X[t,:]   = x[:]
		V[t-1,:] = v[:]

	return X, V

