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

from collections import defaultdict
import math
import time

import numpy as np
import matplotlib.pyplot as plt

from htmresearch_core.experimental import computeGridUniquenessHypercube


def get_me_this_A_your_talking_about(k, m):
    A = np.zeros((m,2,k))

    frac = m//k

    for i in range(k):
        for j in range(0, frac):
            theta = np.random.sample()*np.pi*2.
            a = np.array([np.cos(theta), np.sin(theta)])
            s = np.sqrt(2)**j
            A[i*frac + j, :, i] = a / s
    
    return A

def doRandomModuleExperiment(ms, ks):
  scales = [1.*(math.sqrt(2)**s) for s in xrange(max(ms))]
  phase_resolution = 0.2

  results = {}

  for m in ms:
    for k in ks:
      A_ = get_me_this_A_your_talking_about(k,m)
      result = computeGridUniquenessHypercube(A_, phase_resolution, 0.5)
      results[(m, k)] = result[0]

  return None, results


def experiment1():
  ms = range(3, 6+1)
  ks = range(1, 3)
  numTrials = 10

  allResultsByParams = defaultdict(list)
  for _ in xrange(numTrials):
    A, resultsByParams = doRandomModuleExperiment(ms, ks)
    for params, v in resultsByParams.iteritems():
      allResultsByParams[params].append(v)

  meanResultByParams = {}
  for params, listOfResults in allResultsByParams.iteritems():
    meanResultByParams[params] = (sum(listOfResults) / len(listOfResults))

  timestamp = time.strftime("%Y%m%d-%H%M%S")

  # Diameter plot
  plt.figure()
  for m in ms:
    x = []
    y = []
    for k in ks:
      x.append(k)
      y.append(meanResultByParams[(m,k)])
    plt.plot(x, y, marker='o')
  plt.yscale('log')
  plt.xticks(ks)
  plt.xlabel("Number of dimensions")
  plt.ylabel("Diameter of unique hypercube")
  plt.legend(["{} module{}".format(m, "" if m == 0 else "s")
              for m in ms])
  filename = "results/Diameter_%s.pdf" % timestamp
  print "Saving", filename
  plt.savefig(filename)

  # Volume plot
  plt.figure()
  for m in ms:
    x = []
    y = []
    for k in ks:
      x.append(k)
      y.append(math.pow(meanResultByParams[(m,k)], k))
    plt.plot(x, y, marker='o')
  plt.yscale('log')
  plt.xticks(ks)
  plt.xlabel("Number of dimensions")
  plt.ylabel("Volume of unique hypercube")
  plt.legend(["{} module{}".format(m + 1, "" if m == 0 else "s")
              for m in ms])
  filename = "results/Volume_%s.pdf" % timestamp
  print "Saving", filename
  plt.savefig(filename)


if __name__ == "__main__":
  experiment1()
