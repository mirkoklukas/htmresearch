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
from htmresearch.support.gridcells.random_walk import smooth_walk
from htmresearch.support.gridcells.utils import map_to_quotient, hexagonal_basis, compute_grid, Lp_dist
import numpy as np
from scipy.stats import entropy



def shearing_trial(bins, theta, phi, scale, X, noisyX):
 	
	B  = scale*hexagonal_basis(theta, phi)
 	v  = np.array([0.,0.])
	Z  = map_to_quotient(X, B, v)
	Z_ = map_to_quotient(noisyX, B, v)
	# Z = np.dot(Z, B.T)
	# Z_ = np.dot(Z_, B.T)

	stab = stability(Z,Z_)

	counts, _, _ = np.histogram2d(Z[:,0], Z[:,1], bins=bins);
	p = counts/np.sum(counts)
	h = entropy(p.reshape(-1), base=2)

	return (h, p, stab)


def stability(X, noisyX):
	dist = Lp_dist(X, noisyX, p=2)
	stab = - np.mean(dist)

	return stab


def run_shearing_experiment(bins, thetas, phis, scales, X, noisyX):
	count = 0
	print "Numer of trials: {}".format(np.product((len(thetas), len(phis), len(scales))))
	h  = np.zeros((len(thetas), len(phis), len(scales)))
	st = np.zeros((len(thetas), len(phis), len(scales)))
	p  = np.zeros((len(thetas), len(phis), len(scales), bins, bins))
	for i_theta, theta in enumerate(thetas):
		for i_phi, phi in enumerate(phis):
			for i_scale, scale in enumerate(scales):
					h_, p_, s_ = shearing_trial(bins, theta, phi, scale, X, noisyX)

					h[ i_theta, i_phi, i_scale] = h_
					p[ i_theta, i_phi, i_scale] = p_[:]
					st[i_theta, i_phi, i_scale] = s_

					count += 1

	return h, p, st





