import numpy as np



class ContextLayer(object):

    def __init__(self, layer_shape, module_shape):
        assert module_shape[0] == module_shape[1], "Check module dimensions, we want a square shape..."
        assert np.prod(module_shape) == np.prod(layer_shape)

        self.layer_shape  = layer_shape
        self.module_shape = module_shape

        n, d, m   = self.shape
        self.perm = np.random.permutation(n**2)
        self.perm_inv = np.zeros(n**2).astype(int)
        for i in range(n**2):
            self.perm_inv[self.perm[i]] = i

        self.state = np.zeros(n**2)

    def clear(self):
        self.state[:] = 0

    @property
    def shape(self):
        return self.module_shape[0], self.layer_shape[0], self.layer_shape[1]

    @property
    def module(self):
        n, _, _   = self.shape
        perm = self.perm
        return self.state[perm].reshape((n, n))

    @property
    def layer(self):
        _, d, m   = self.shape
        return self.state.reshape((d, m))

    def explore(self, a, mentally=False):
        n, _, _   = self.shape
        perm_inv = self.perm_inv
        
        C  = self.module
        C_ = np.zeros((n,n))
        for x0 in range(n):
            for x1 in range(n):
                y0 =(x0 + a[0])%n
                y1 =(x1 + a[1])%n
                C_[y0,y1] += C[x0, x1]
        
        C_ = np.clip(C_, 0, 1)

        if mentally == False:
            self.state = C_.reshape(-1)[perm_inv]
            return self.layer
        else:
            return C_.reshape(-1)[perm_inv].reshape(self.layer_shape)

    def intersect(self, X):
        assert X.shape == self.layer_shape

        self.state *= X.reshape(-1)
        self.state = np.clip(self.state, 0,1)

        return self.layer

    def add(self, X):
        """
        Extend the current context 
        (or state respectively)
        """
        assert X.shape == self.layer_shape

        self.state += X.reshape(-1)
        self.state = np.clip(self.state, 0,1)

        return self.layer

    def extend(self, a, X, intersect=False):
        """Extend the current context"""

        self.explore(a)
        if not intersect:
            self.add(X)
        else:
            self.intersect(X)

        return self.layer





