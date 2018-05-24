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


def exp_i(theta):
	return np.array([np.cos(theta), np.sin(theta)])

def hexagonal_basis(theta, psi=np.pi/3.0):
	return np.array([
		[np.cos(theta), np.cos(theta + psi)],
		[np.sin(theta), np.sin(theta + psi)]
	])

def Lp_dist(X,Y, p=2):
	assert(p<=2)
	
	x  = X[:,0]
	x_ = Y[:,0]
	y  = X[:,1]
	y_ = Y[:,1]

	dx = np.minimum(np.abs(x - x_), np.abs(x_ - x))%1
	dy = np.minimum(np.abs(y - y_), np.abs(y_ - y))%1
	if p==1:
		return dx + dy
	elif p==2:
		return np.sqrt(dx**2 + dy**2)


def map_to_quotient(X, B, v=np.array([0.,0.])):
	X_ = X - v
	Y = np.dot(X_, np.linalg.inv(B).T)
	Y %= 1
	return Y

	

def compute_grid(B, r=10):
	L = np.array([ x*B[:,0] + y*B[:,1] for x in range(-r,r+1) for y in range(-r,r+1)])
	return L





def cross_correlate(left, right):
	nrows, ncols = left.shape
	di_cap = nrows - 1
	dj_cap = ncols - 1
	ci = di_cap - 1
	cj = dj_cap - 1
	corr = np.zeros((2*di_cap - 1, 2*dj_cap - 1), dtype="float")

	for di in xrange(di_cap):
		for dj in xrange(dj_cap):
			  # Spatial lag: up, left
			  corr[ci - di, cj - dj] = np.corrcoef(
			    left[di:, dj:].flat,
			    right[:nrows-di, :ncols-dj].flat)[0,1]
			  # Spatial lag: up, right
			  corr[ci - di, cj + dj] = np.corrcoef(
			    left[di:, :ncols-dj].flat,
			    right[:nrows-di, dj:].flat)[0,1]
			  # Spatial lag: down, left
			  corr[ci + di, cj - dj] = np.corrcoef(
			    left[:nrows-di, dj:].flat,
			    right[di:, :ncols-dj].flat)[0,1]
			  # Down right
			  corr[ci + di, cj + dj] = np.corrcoef(
			    left[:nrows-di, :ncols-dj].flat,
			    right[di:, dj:].flat)[0,1]

	return corr




