# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

from nupic.algorithms.spatial_pooler import SpatialPooler 
import numpy as np
import numpy
from nupic.bindings.math import GetNTAReal
realDType = GetNTAReal()
PERMANENCE_EPSILON = 0.000001


class GridPooler(SpatialPooler):
  """
  An experimental spatial pooler implementation
  with learned lateral inhibitory connections.
  """
  def __init__(self, num_velocity, **spArgs):

    spArgs["inputDimensions"] = [spArgs["columnDimensions"][0] * num_velocity,1]
    super(GridPooler, self).__init__(**spArgs)

    self._numVelocity = num_velocity
    self.codeWeight   = self._numActiveColumnsPerInhArea
    self.sparsity     = float(self.codeWeight)/float(self._numColumns)


    # Varibale to store average pairwise activities
    s = self.sparsity


  def connect_pair(self,v,x,x_):
    vx      = np.zeros((self._numVelocity, self._numColumns))
    vx[v,:] = x[:]
    vx = vx.reshape(-1)

    activeColumns = np.where(x_==1)[0]
    self._adaptSynapses(vx, activeColumns)
    
    if v !=0:
      self._updateDutyCycles(self._overlaps, activeColumns)
      self._bumpUpWeakColumns()

      self._updateBoostFactors()





  def fit_v(self, v, x):
    """
    This is the primary public method of the LateralPooler class. This
    function takes a input vector and outputs the indices of the active columns.
    If 'learn' is set to True, this method also updates the permanences of the
    columns and their lateral inhibitory connection weights.
    """


    vx      = np.zeros((self._numVelocity, self._numColumns))
    vx[v,:] = x[:]
    vx = vx.reshape(-1)
    x_      = np.zeros(self._numColumns)

    if v==0:
      self.connect_pair(0,x,x)

    else:
      self._overlaps = self._calculateOverlap(vx)

      # Apply boosting when learning is on
      if True:
        self._boostedOverlaps = self._boostFactors * self._overlaps
      else:
        self._boostedOverlaps = self._overlaps

      # Apply inhibition to determine the winning columns
      activeColumns = self._inhibitColumns(self._boostedOverlaps)
      x_[activeColumns] = 1.0


      self.connect_pair(v,x,x_)
      self.connect_pair(-v,x_,x)

    return x_


  def encode(self, X, applyLateralInhibition=True):
    """
    This method encodes a batch of input vectors.
    Note the inputs are assumed to be given as the 
    columns of the matrix X (not the rows).
    """
    d = X.shape[1]
    n = self._numColumns
    Y = np.zeros((n,d))
    for t in range(d):
        self.compute(X[:,t], False, Y[:,t], applyLateralInhibition)
        
    return Y


  @property
  def feedforward(self):
    """
    Soon to be depriciated.
    Needed to make the SP implementation compatible 
    with some older code.
    """
    m = self._numInputs
    n = self._numColumns
    W = np.zeros((n, m))
    for i in range(self._numColumns):
        self.getPermanence(i, W[i, :])

    return W

  @property
  def code_weight(self):
    """
    Soon to be depriciated.
    Needed to make the SP implementation compatible 
    with some older code.
    """
    return self._numActiveColumnsPerInhArea


  @property
  def smoothing_period(self):
    """
    Soon to be depriciated.
    Needed to make the SP implementation compatible 
    with some older code.
    """
    return self._dutyCyclePeriod



  # def _updateBoostFactorsGlobal(self):
  #   """
  #   Update boost factors when global inhibition is used
  #   """
  #   # When global inhibition is enabled, the target activation level is
  #   # the sparsity of the spatial pooler
  #   if (self._localAreaDensity > 0):
  #     targetDensity = self._localAreaDensity
  #   else:
  #     inhibitionArea = ((2 * self._inhibitionRadius + 1)
  #                       ** self._columnDimensions.size)
  #     inhibitionArea = min(self._numColumns, inhibitionArea)
  #     targetDensity = float(self._numActiveColumnsPerInhArea) / inhibitionArea
  #     targetDensity = min(targetDensity, 0.5)


  #   # Usual definition
  #   self._beta = (targetDensity - self._activeDutyCycles)

  #   # Experimental setting
  #   # self._beta += 0.001*(targetDensity - self._activeDutyCycles)
    
  #   self._boostFactors = np.exp(self._beta * self._boostStrength)






