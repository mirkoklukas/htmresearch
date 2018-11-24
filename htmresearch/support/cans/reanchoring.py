import numpy as np


def relu(x):
    return np.maximum(x, 0.)


def W_zero(x):
    a          = 1.
    lambda_net = 4.0
    beta       = 3.0 / lambda_net**2
    gamma      = 1.05 * beta
    
    x_length_squared = x**2
    
    return 1.*(a*np.exp(-gamma*x_length_squared) - np.exp(-beta*x_length_squared))


def create_recurrent_weights_1d(n, d):
    X = np.linspace(0., d, num=n, endpoint=False)
    J = W_zero
    # D = X.reshape((n,1)) - X.reshape((1,n))
    # D = np.absolute(D)
    # D = np.minimum(d - D, D)

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
                dij = np.absolute(X[i] - X[j])
                D[i,j] = np.minimum(d - dij, dij )


    W = J(D)
    np.fill_diagonal(W, 0.0)
    return W


def create_recurrent_weights_2d(n, d):
    X = np.mgrid[0:d:d/n, 0:d:d/n].reshape(2,-1).T

    J = W_zero
    # D = X.reshape((n,1)) - X.reshape((1,n))
    # D = np.absolute(D)
    # D = np.minimum(d - D, D)

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
                dij = np.absolute(X[i] - X[j])
                D[i,j] = np.minimum(d - dij, dij )


    W = J(D)
    np.fill_diagonal(W, 0.0)
    return W


class GridModule1d(object):
    def __init__(self, num_cells, diameter, vel_gain):
        self.num_cells = num_cells
        self.diameter  = diameter
        self.recurrent_weights = create_recurrent_weights_1d(num_cells, diameter)
        self.s = np.zeros(num_cells)
        self.vel_gain = vel_gain

    @property
    def W(self):
        return self.recurrent_weights

    @property
    def n(self):
        return self.num_cells
    @property
    def d(self):
        return self.diameter


    def evolve(self, b=0.001, v=np.zeros(2), dt=0.0005, tau=0.03, f=relu):
        # print b
        b_  = np.zeros(self.n) + b
        b_ += np.roll(self.s, 10)*v[1]*self.vel_gain + np.roll(self.s, -10)*v[0]*self.vel_gain

        Ws  = np.dot(self.W, self.s)
        ds  = ( f(Ws + b_ ) - self.s/tau )*dt
        self.s = self.s + ds

        return self.s


class MEC(object):
    def __init__(self, num_modules, cells_per_module, diameter_of_modules, vel_gains):
        self.modules = []
        for i in range(num_modules):
            self.modules.append(GridModule1d(cells_per_module, diameter=diameter_of_modules, vel_gain=vel_gains[i]))


        







