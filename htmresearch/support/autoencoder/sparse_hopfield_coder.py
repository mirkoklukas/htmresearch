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




class SparseHopfieldCoder(object):
    """
    An naive implementation of the 
    dynamic system described in 

        Rozell, Johnson, Baraniuk & Olshausen: "Sparse Coding via Thresholding and Local Competition in Neural Circuits"
        Neural Computation
        2526-2563, 2008.
    
    """
    def __init__(self, 
                num_inputs=784, 
                num_outputs=100,  
                learning_rate=0.1):


        n = num_outputs
        m = num_inputs

        self.num_outputs = n
        self.num_inputs  = m

        


        self.learning_rate = learning_rate
        

        
        eps = tf.placeholder(tf.float32, shape=())
        lam = tf.placeholder(tf.float32, shape=())

        Phi = tf.Variable(tf.random_normal([m,n], dtype=tf.float32), name="codebook")
        S   = tf.placeholder(tf.float32, shape=[m,1], name="stimulus")

        

        # Create variables for simulation state
        U  = tf.get_variable("U", [n,1], dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
        Ut = tf.get_variable("Ut", [n,1], dtype=tf.float32, initializer=tf.zeros_initializer,trainable=False)
        A  = tf.get_variable("A", [n,1], dtype=tf.float32, initializer=tf.zeros_initializer,trainable=False)


        B = tf.matmul(Phi, S, transpose_a=True, name="bias")
        G = tf.matmul(Phi,Phi, transpose_a=True, name="Gram")
        A = tf.nn.relu(U - lam)

        # Discretized PDE update rules
        U_  = U + eps*Ut
        Ut_ = B - U - tf.matmul(G, A) + A

        # Operation to update the state
        self.step = tf.group(
          U.assign(U_),
          Ut.assign(Ut_))



        self.A_    = tf.placeholder(tf.float32, shape=[n,1])
        self.S_hat = tf.matmul(Phi, self.A_)


        self.lam = lam
        self.eps = eps
        self.U = U
        self.Ut = Ut
        self.Phi = Phi
        self.S = S
        self.A = A



        self.loss  = tf.reduce_mean(tf.square(tf.subtract(self.S, self.S_hat))) 
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        

    def encode(self, S, N=100, lam=1.0):


        init_u = tf.variables_initializer([self.U, self.Ut])
        self.sess.run([init_u])
        # tf.global_variables_initializer().run()
        # Run 1000 steps of PDE
        U = np.zeros((N, self.num_outputs))
        A = np.zeros((N, self.num_outputs))

        for t in range(N):
            # Step simulation
            _, u, ut, a = self.sess.run(fetches   = [self.step, self.U, self.Ut, self.A], 
                                        feed_dict = {self.eps: 0.01, self.S: S, self.lam: lam})

            U[t,:] = u[:,0]
            A[t,:] = a[:,0]

        return A


    def fit(self, S, N=500, lam=1.0):
        d      = len(S)
        Y      = np.zeros((d, self.num_outputs))
        S_hat  = np.zeros((d, self.num_inputs))
        losses = np.zeros(d)

        for i in range(d):
            y = self.encode(S[[i]].T,  N, lam)
            Y[i,:] = y[-1,:]
            _, loss, s_hat = self.sess.run([self.train_step, self.loss, self.S_hat], feed_dict={self.eps: 0.01, self.S: S[[i]].T,self.A_: y[[-1]].T, self.lam: lam})
            losses[i] = loss
            S_hat[i,:] = s_hat[:,0]

        return losses, Y, S_hat













