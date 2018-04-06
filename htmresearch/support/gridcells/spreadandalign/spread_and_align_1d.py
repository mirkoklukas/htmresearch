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
import tensorflow as tf


class SpreadAndAlign1dModel(object):
  """
  Straightforward implementation of a rnn model whose
  hidden nodes show grid-cell-like firing activity...
  ```
  data       = load_data()
  parameters = load_parameters()
  model      = SpreadAndAlignModel(parameters)
  model.fit(data)

  ```

  model_parameters = {
    'num_place_cells'   : 400,
    'num_grid_cells'    : 25,
    'num_velocity_cells': 2,
    'prediction_loss_weight': 1.0,
    'sparseness_loss_weight': 3.0,
    'spread_loss_weight'    : 5.0,
    'code_weight'  : 2.0,
    'spread_weight': 1.0,
    'learning_rate': 0.2
  }

  """
  def __init__(self, parameters):
    self.parameters = parameters
    
    self.learning_rate      = parameters["learning_rate"]
    self.num_grid_cells     = parameters["num_grid_cells"]
    self.num_velocity_cells = parameters["num_velocity_cells"]
    num_h = self.num_grid_cells
    num_v = self.num_velocity_cells


    lim_hh=np.sqrt(6. / sum([num_h,num_h]))
    lim_hv=np.sqrt(6. / sum([num_h,num_v]))


    W_hh   = tf.Variable(tf.random_uniform([num_h,num_h], minval= -lim_hh, maxval=lim_hh,dtype=tf.float64), name="Rec_weights")
    W_hv   = tf.Variable(tf.random_uniform([num_h,num_v], minval= -lim_hv, maxval=lim_hv,dtype=tf.float64), name="Vel_weights")
    b      = tf.Variable(tf.zeros([num_h,1], dtype=tf.float64), name="bias")
    v      = tf.placeholder(tf.float64, shape=[num_v, 1], name="v")
    h_zero = tf.placeholder(tf.float64, shape=[num_h, 1], name="h_zero")

    self.b = b
    self.v = v
    self.h_zero = h_zero
    self.W_hh = W_hh
    self.W_hv = W_hv

    self.sess = tf.Session()
    init = tf.global_variables_initializer()
    self.sess.run(init)


  @property
  def fwd_weights(self):
    """
    """
    if type(self.W_hx) == tf.layers.Dense:
      W = self.W_hx.weights[0].value().eval(session=self.sess).T
    else:
      W = self.W_hx.eval(session=self.sess)
    return W


  def fit(self, data):
    """
    """

    V = data["V"].reshape(-1)
    d = V.shape[0]

    num_h  = self.num_grid_cells
    h_prev = np.zeros((num_h, 1))
    losses = []
    h_zero = self.h_zero
    W_hh   = self.W_hh
    W_hv   = self.W_hv
    b      = self.b

    softmax = tf.nn.softmax
    sigmoid = tf.nn.sigmoid
    activation = sigmoid

    h_ = activation( tf.matmul(W_hh, h_zero) + tf.matmul( W_hv, V[0].reshape((1,1)) )  + b )


    loss    = tf.reduce_sum( h_ * tf.log(h_) )
    for t in range(1,d):
        h_ = activation( tf.matmul(W_hh, h_) + tf.matmul( W_hv, V[t].reshape((1,1)) )  + b )
        loss += tf.reduce_sum( h_ * tf.log(h_) )

    
    diff = tf.subtract(h_zero, h_)
    rec  = tf.tensordot(diff, diff, axes=2)
    loss = loss + rec
    

    # optimizer = tf.train.AdamOptimizer()
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    train_step = optimizer.minimize(loss)

    feed_dict = {
      h_zero: data["h_zero"]
    }

    _ = self.sess.run(  fetches   = [train_step], 
                        feed_dict = feed_dict)


  @property
  def hh_weights(self):
    """
    """
    if type(self.W_hh) == tf.layers.Dense:
      W = self.W_hh.weights[0].value().eval(session=self.sess).T
    else:
      W = self.W_hh.eval(session=self.sess)
    return W

  @property
  def hv_weights(self):
    """
    """
    if type(self.W_hv) == tf.layers.Dense:
      W = self.W_hv.weights[0].value().eval(session=self.sess).T
    else:
      W = self.W_hv.eval(session=self.sess)
    return W


  @property
  def bias(self):
    """
    """

    b = self.b.eval(session=self.sess)
    return b












