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
from scipy.special import expit as sigmoid




class LateralKSparseAutoencoder(object):
    """
    An experimental implementation of a k-sparse autoencoder
    with learned lateral inhibitory connections.
    """
    def __init__(self, num_inputs, num_outputs, code_weight, beta, learning_rate, enable_boosting=True, enforce_binary_output=False, enforce_code_weight=True, with_lateral=True):

        n, m = num_outputs, num_inputs

        self.num_inputs    = num_inputs
        self.num_outputs   = num_outputs
        self.code_weight   = code_weight
        self.sparsity      = float(code_weight)/float(num_outputs)
        self.beta          = beta 
        self.learning_rate         = learning_rate
        self.learning_rate_lateral = learning_rate

        self.num_trained = 0


        self.enforce_binary_output = enforce_binary_output
        self.enable_boosting = enable_boosting
        self.enforceDesiredWeight = enforce_code_weight
        self.enforce_code_weight  = enforce_code_weight
        self.with_lateral = with_lateral


        self.lateralConnections = np.ones((n,n))/float(n-1)
        np.fill_diagonal(self.lateralConnections, 0.0)

        
        self.mean_activity    = np.ones(n)*self.sparsity
        self.avgActivityPairs = np.ones((n,n))*(self.sparsity**2)


        self.x  = tf.placeholder(tf.float64, shape=[m, 1], name="x")
        # self.e  = tf.placeholder(tf.float64, shape=[n, 1], name="e")
        self.y_ = tf.placeholder(tf.float64, shape=[n, 1], name="y_binary")

        self.W = tf.Variable(tf.random_normal([n,m], dtype=tf.float64), name="Fwd_weights")

        self.e = tf.placeholder(tf.float64, shape=[n, 1], name="e")


        y = self.e*self.y_


        x_hat = tf.matmul(self.W, y, transpose_a=True)

        
        # Reconstruction error
        self.loss  = tf.reduce_mean(tf.square( tf.subtract(self.x, x_hat))) 
        # self.loss  = tf.reduce_mean(tf.abs( tf.subtract(self.x, x_hat))) 
        # self.loss = tf.reduce_mean( tf.square(  tf.subtract(x,tf.sigmoid(x_hat)) ))
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_hat))

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)



    def fit(self, X):

        d = len(X)
        n = self.num_outputs

        losses = np.zeros(d)
        E      = np.zeros((d, n))
        Y      = np.zeros((d, n))
        k = self.code_weight
        for t in range(d):

            b = self._get_boostfactor()
            # x = X[t].reshape((-1,1))


            e = np.dot(self.weights, X[[t]].T)

            if self.enable_boosting:
                score = b*e
            else:
                score = e


            if self.with_lateral:
                Y[t,:] = self._inhibitColumnsWithLateral(score, self.lateralConnections).reshape(-1)

            else:
                sortedIndices = np.argsort( score.reshape(-1), kind='mergesort')[::-1]
                Y[t,sortedIndices[:k]] = 1.0


            feed_dict = {self.x : X[[t]].T, self.e: e, self.y_: Y[[t],:].T}

            _, loss, e = self.sess.run([self.train_step, self.loss, self.e], feed_dict = feed_dict)


            self._updateAvgActivityPairs(Y[t])
            # epsilon = self.learning_rate_lateral
            epsilon = 1.0
            self._updateLateralConnections(epsilon, self.avgActivityPairs)
            self._update_mean_activity(Y[t])

            E[t,:] = e[:,0]

            losses[t] = loss

            if self.enforce_binary_output==False:
                Y[t] *= e[:,0]


        self.num_trained += d

        return Y, E, losses



    def encode(self, X, with_boosting=False, with_lateral=True, enforce_binary_output=True):
        d = len(X)
        Y = np.zeros((d, self.num_outputs))
        E = np.zeros((d, self.num_outputs))
        b = self._get_boostfactor().reshape(-1)
        W = self.weights
        k = self.code_weight
        for t in range(d):
            x = X[t].reshape((-1,1))
            e = np.dot(self.weights, x).reshape(-1)
            E[t,:] = e

            if with_boosting==True:
                score = b*e
            else:
                score  = e

            if with_lateral:
                Y[t,:] = self._inhibitColumnsWithLateral(score, self.lateralConnections).reshape(-1)

            else:
                sortedIndices = np.argsort( score, kind='mergesort')[::-1]
                Y[[t],sortedIndices[:k]] = 1.0

            if enforce_binary_output==False:
                Y[t] *= e 

        return Y



    def _get_boostfactor(self, strength=100):
        alpha = np.clip(self.mean_activity,0.000001,1)
        # boo = (1./alpha).reshape((-1,1))
        boo = np.exp( - strength*self.mean_activity ).reshape((-1,1))
        return boo


    def _inhibitColumnsWithLateral(self, overlaps, lateralConnections):
        """
        Performs an experimentatl local inhibition. Local inhibition is 
        iteratively performed on a column by column basis.
        """
        n = self.num_outputs
        overlaps = overlaps.reshape(-1)
        y   = np.zeros(n)
        s   = self.sparsity
        L   = lateralConnections

        desiredWeight = self.code_weight
        inhSignal     = np.zeros(n)
        sortedIndices = np.argsort(overlaps, kind='mergesort')[::-1]

        currentWeight = 0
        for i in sortedIndices:

          inhTooStrong = ( inhSignal[i] >= s )

          if not inhTooStrong:
            y[i]              = 1.
            currentWeight    += 1
            inhSignal[:]     += L[i,:]

          if self.enforceDesiredWeight and currentWeight == desiredWeight:
            break

        # activeColumns = np.where(y==1.0)[0]

        return y.reshape((-1,1))


    def _updateAvgActivityPairs(self, activeArray):
        """
        Updates the average firing activity of pairs of 
        columns.
        """
        n = self.num_outputs
        m = self.num_inputs
        Y    = activeArray.reshape((n,1))
        beta = self.beta

        Q = np.dot(Y, Y.T) 

        self.avgActivityPairs = (1.0-beta)*self.avgActivityPairs + beta*Q

    def _update_mean_activity(self, y):
        b = self.beta
        self.mean_activity = (1.0-b)*self.mean_activity + b*y

    def _updateLateralConnections(self, epsilon, avgActivityPairs):
        """
        Sets the weights of the lateral connections based on 
        average pairwise activity of the SP's columns. Intuitively: The more 
        two columns fire together on average the stronger the inhibitory
        connection gets. 
        """
        oldL = self.lateralConnections
        newL = avgActivityPairs.copy()
        np.fill_diagonal(newL, 0.0)
        newL = newL/np.sum(newL, axis=1, keepdims=True)

        self.lateralConnections[:,:] = (1.0 - epsilon)*oldL + epsilon*newL
        np.fill_diagonal(self.lateralConnections, 0.0)


    @property
    def weights(self):
        W = self.W.eval(session=self.sess)
        return W


    def __str__(self):
        summary = "\n**Lateral k-sparse autoencoder:**"\
                  "\n------------------"\
                  "\nNumber of inputs (m):\t {self.num_inputs}"\
                  "\nNumber of outputs (n):\t {self.num_outputs}"\
                  "\nCode weight (k):\t {self.code_weight}"\
                  "\nSparsity (k/n):\t\t {self.sparsity}"\
                  "\nBeta:\t\t\t {self.beta}"\
                  "\nLearning rate:\t\t {self.learning_rate}"\
                  "\nMin/Max weights :\t {minW:+.2f}  |  {maxW:+.2f}"\
                  "\nBinary outupt:\t\t {self.enforce_binary_output}"\
                  "\nBoosting:\t\t {self.enable_boosting}"\
                  "\nLateral:\t\t {self.with_lateral}"\
                  "\nEnforce code weight:\t {self.enforce_code_weight}"\
                  "\n------------------".format(self=self, minW = np.amin(self.weights), maxW = np.amax(self.weights))
                  
        return summary

