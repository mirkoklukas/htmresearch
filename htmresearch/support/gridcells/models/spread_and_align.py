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


class SpreadAndAlignModel(object):
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
    self.num_place_cells    = parameters["num_place_cells"]
    self.num_grid_cells     = parameters["num_grid_cells"]
    self.num_velocity_cells = parameters["num_velocity_cells"]

    self.beta = parameters["beta"]
    self.mean_activity = np.zeros((self.num_grid_cells,1))

    # 
    # The Graph
    # 
    num_h, num_x, num_v = self.shape 
    x    = tf.placeholder(tf.float32, shape=[num_x, 1], name="x")
    v_   = tf.placeholder(tf.float32, shape=[num_v, 1], name="v_")
    h_   = tf.placeholder(tf.float32, shape=[num_h, 1], name="h_")

    boost= tf.placeholder(tf.float32, shape=[num_h, 1], name="boost")
    self.boost = boost

    lim_hx=np.sqrt(6. / sum([num_h,num_x]))
    lim_hh=np.sqrt(6. / sum([num_h,num_h]))
    lim_hv=np.sqrt(6. / sum([num_h,num_v]))



    W_hx = tf.Variable(tf.random_uniform([num_h,num_x], minval= -lim_hx, maxval=lim_hx, dtype=tf.float32), name="Fwd_weights")
    W_hh = tf.Variable(tf.random_uniform([num_h,num_h], minval= -lim_hh, maxval=lim_hh, dtype=tf.float32), name="Rec_weights")
    W_hv = tf.Variable(tf.random_uniform([num_h,num_v], minval= -lim_hv, maxval=lim_hv, dtype=tf.float32), name="Vel_weights")

    # W_hx = tf.Variable(tf.random_uniform([num_h,num_x], minval= 0, maxval=1, dtype=tf.float32), name="Fwd_weights")
    # W_hh = tf.Variable(tf.random_uniform([num_h,num_h], minval= 0, maxval=1, dtype=tf.float32), name="Rec_weights")
    # W_hv = tf.Variable(tf.random_uniform([num_h,num_v], minval= 0, maxval=1, dtype=tf.float32), name="Vel_weights")
    b_x = tf.Variable(tf.zeros([num_h,1]), name="biases")
    b_h = tf.Variable(tf.zeros([num_h,1]), name="biases")

    
    softmax = tf.nn.softmax
    sigmoid = tf.nn.sigmoid
    # h_fwd = tf.transpose(softmax(tf.transpose( tf.matmul( W_hx, x ) + b_x ) ))
    # h_hat = tf.transpose(softmax(tf.transpose( tf.matmul( W_hh, h_) + tf.matmul( W_hv, v_ )  + b_h ) ))

    z = tf.matmul( W_hx, x ) + b_x
    h_fwd = boost * sigmoid( z  ) 
    h_hat = sigmoid( tf.matmul( W_hh, h_) + tf.matmul( W_hv, v_ )  + b_x ) 


    # drop = tf.layers.dropout(
    #     inputs=h_hat,
    #     rate=0.5,
    #     noise_shape=None,
    #     seed=None,
    #     training=False,
    #     name=None
    # )
    # 
    # Option b
    # 
    # W_hx = tf.layers.Dense(units=num_h, activation=tf.nn.sigmoid, name="Fwd_weights", trainable=True)
    # W_hhv = tf.layers.Dense(units=num_h, activation=tf.nn.sigmoid, name="Rec_weights", trainable=True)
    # h_conc_v = tf.concat(values=[h_,10.0*v_], axis=0, name="h_concat_v")
    # h_fwd = tf.transpose(W_hx(tf.transpose(x)))
    # h_hat = tf.transpose(W_hhv(tf.transpose(h_conc_v)))

    print x.shape, h_fwd.shape

    self.x  = x
    self.v_ = v_
    self.h_ = h_
    self.h_fwd = h_fwd
    self.h_hat = h_hat
    self.W_hx = W_hx

    # 
    # The Loss
    # 
    self.loss_fn = self._loss_fn
    learning_rate = parameters["learning_rate"]

    if parameters["optimizer"] == "adam":
        optimizer = tf.train.AdamOptimizer()
    elif parameters["optimizer"] == "gradient":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    self.train_step = optimizer.minimize(self.loss_fn)


    # 
    # The Session
    # 
    self.sess = tf.Session()
    init = tf.global_variables_initializer()
    self.sess.run(init)


  @property
  def _loss_fn(self):
    parameters = self.parameters
    h_fwd      = self.h_fwd
    h_hat      = self.h_hat
    h_         = self.h_
    v_         = self.v_
    # Prediction loss:
    # 
    #    < \hat h - h_t , \hat h - h_t >  =  | \hat h - h_t |^2
    # 
    diff =  tf.subtract(h_fwd, h_hat)
    prediction_loss = tf.tensordot(diff, diff, axes=2)
    # prediction_loss = - tf.reduce_sum( h_hat * tf.log(h_fwd) )
    # prediction_loss =   tf.reduce_sum(h_fwd * tf.log(tf.divide(h_fwd, h_hat)) )

    # Sparse penalty:
    #     
    #     ( < h_t , h_t > - "code weight" )^2 
    # 
    code_weight     = parameters["code_weight"]
    h_fwd_squared   = tf.tensordot( h_fwd, h_fwd, axes=2)
    sparseness_loss = tf.pow( h_fwd_squared - code_weight, 2)
    # sparseness_loss = - tf.reduce_sum( h_fwd * tf.log(h_fwd) )


    # Variability (or spread) loss:
    #     
    #     ( < h_t , h_{t+1} >  - "desired spread" )^2
    # 
    spread_weight = parameters["spread_weight"]
    # h_squared = tf.tensordot( h_, h_, axes=2)
    # normalizer = h_squared*h_fwd_squared
    # v_sq = tf.tensordot( v_, v_, axes=2)
    # spread_loss   = v_sq*tf.pow(  tf.tensordot( h_, h_fwd, axes=2) - normalizer*v_sq*spread_weight, 2)
    # diff =  tf.subtract(h_fwd, h_)
    # prediction_loss = tf.tensordot(diff, diff, axes=2) - normalizer*normalizer*v_sq
    spread_loss   = tf.pow(tf.reduce_sum( tf.pow(tf.subtract(h_, h_fwd), 2)) - spread_weight, 2)
    # spread_loss = tf.reduce_sum( tf.tensordot(h_fwd , tf.log(tf.divide(h_fwd, h_)), axes=2) ) - 1
    # spread_loss = - tf.reduce_sum( h_fwd * tf.log(tf.divide(h_fwd, h_)) )*(10.0*spread_weight)
    # spread_loss = tf.reduce_sum( h_fwd * tf.log(h_) )*spread_weight


    a = parameters["prediction_loss_weight"]
    b = parameters["sparseness_loss_weight"]
    c = parameters["spread_loss_weight"]
    loss = a*prediction_loss + b*sparseness_loss + c*spread_loss

    return loss


  @property
  def shape(self):
    """
    """
    return (self.num_grid_cells, self.num_place_cells, self.num_velocity_cells)

  @property
  def placeholders(self):
    """
    """
    return (self.x, self.v_, self.h_)

  @property
  def fwd_weights(self):
    """
    """
    if type(self.W_hx) == tf.layers.Dense:
      W = self.W_hx.weights[0].value().eval(session=self.sess).T
    else:
      W = self.W_hx.eval(session=self.sess)
    return W

  def get_boost(self):
    """
    """
    return np.exp( - 100*self.mean_activity )

  def fit(self, data):
    """
    """
    X = data["X"]
    V = data["V"]
    d = X.shape[0]

    beta = self.beta
    num_h  = self.num_grid_cells
    h_prev = np.zeros((num_h, 1))
    losses = []
    for t in range(d):

      feed_dict = {
          self.x : X[[t]].T,
          self.v_: V[[t]].T,
          self.h_: h_prev,
          self.boost: np.exp( - 100*self.mean_activity )
      }
      _, loss_value, h_prev = self.sess.run([self.train_step, self.loss_fn, self.h_fwd], feed_dict = feed_dict)


      self.mean_activity = (1-beta)*self.mean_activity + beta*h_prev
      

      losses.append(loss_value)

    return losses















