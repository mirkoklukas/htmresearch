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



class kSparseAutoencoder(object):
    """
    A naive implementation of the k-sparse autoencoder described in:

        A. Makhzani and B. Frey, k-sparse autoencoders
        Advances in Neural Information Processing Systems
        pages 2791--2799, 2015.
        
    """
    def __init__(self, num_inputs, num_outputs, code_weight, learning_rate, beta=0.001, train_only_decoder=False):

        n, m = num_outputs, num_inputs

        self.num_inputs    = num_inputs
        self.num_outputs   = num_outputs
        self.code_weight   = code_weight
        self.sparsity      = float(code_weight)/float(num_outputs)
        self.learning_rate = learning_rate
        self.avg_activity  = np.zeros(num_outputs)
        self.beta = beta
        self.train_only_decoder = train_only_decoder
        self.num_trained = 0
        # 
        #   Tensorflow portion of the model
        # 
        self.x        = tf.placeholder(tf.float64, shape=[m, 1], name="x")
        self.hot_topk = tf.placeholder(tf.float64, shape=[n, 1], name="active_units")
        self.score    = tf.placeholder(tf.float64, shape=[n, 1], name="score")

        self.W  = tf.Variable(tf.random_normal([n,m], dtype=tf.float64), name="Fwd_weights")
        self.bn = tf.Variable(0.2*tf.ones([n,1], dtype=tf.float64), name="bn")
        self.bm = tf.Variable(0.2*tf.ones([m,1], dtype=tf.float64), name="bm")

        if self.train_only_decoder == True:
            z = self.score
        else:
            z = tf.matmul(self.W, self.x)

        z_sparse = z*self.hot_topk

        self.x_hat = tf.matmul(self.W, z_sparse, transpose_a=True) 

        self.loss  = tf.reduce_mean(tf.square( tf.subtract(self.x, self.x_hat))) 

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)



    def fit(self, X):

        d = len(X)
        n = self.num_outputs
        k = self.code_weight
        E = np.zeros((d, n))
        Y = np.zeros((d, n))
        losses = []


        for t in range(d):
            x = X[[t]].T
            W = self.weights
            s = np.dot(W, x)
            y = np.zeros((n,1))
            sortedIndices = np.argsort(s[:,0], kind='mergesort')[::-1]
            y[sortedIndices[:k],0] = 1.0

            feed_dict = {
                self.x : x,
                self.hot_topk: y,
                self.score: s
            }

            _, loss = self.sess.run([self.train_step, self.loss], feed_dict = feed_dict)


            Y[t,:] = s[:,0]*y[:,0]
            E[t,:] = s[:,0]

            losses.append(loss)
            self.update_avg_activity(y[:,0])


        self.num_trained += d

        return Y, E, losses


    def encode(self, X):
        W  = self.weights
        k  = self.code_weight 
        S  = np.dot(W,X.T)
        S_ = np.sort(S, axis=0)
        S[np.where(S < S_[[-k],:])] = 0.

        return S.T


    @property
    def weights(self):
        W = self.W.eval(session=self.sess)
        return W

    def update_avg_activity(self, y):
        y_ = (y > 0.0).astype(float) 
        self.avg_activity = (1.0-self.beta)*self.avg_activity + self.beta*y_


    def __str__(self):
        summary = "\n**k-sparse autoencoder:**"\
                  "\n------------------"\
                  "\nNumber of inputs (m):\t {self.num_inputs}"\
                  "\nNumber of outputs (n):\t {self.num_outputs}"\
                  "\nCode weight (k):\t {self.code_weight}"\
                  "\nSparsity (k/n):\t\t {self.sparsity}"\
                  "\nLearning rate:\t\t {self.learning_rate}"\
                  "\nMin/Max weights :\t {minW:+.2f}  |  {maxW:+.2f}"\
                  "\nTrain only decoder:\t {self.train_only_decoder}"\
                  "\n------------------".format(self=self, minW = np.amin(self.weights), maxW = np.amax(self.weights))
                  
        return summary

