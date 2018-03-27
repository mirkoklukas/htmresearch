import numpy as np

from scipy.stats import entropy




class ContextLayer(object):

    def __init__(self, layer_shape, module_shape, action_map, max_activity=10000):
        assert module_shape[0] == module_shape[1], "Check module dimensions, we want a square shape..."
        assert np.prod(module_shape) == np.prod(layer_shape), "Check layer dimensions..."

        self.layer_shape  = layer_shape
        self.module_shape = module_shape
        self.max_activity = max_activity

        self.action_map = action_map

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

    def _explore(self, a, mentally=False):
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

    def explore(self, a, mentally=False):
        a = np.dot(a, self.action_map)
        return self._explore(a, mentally)

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

    def extend(self, a, X):
        """Extend the current context"""

        a = np.dot(a, self.action_map)

        self._explore(a)

        n, _, _   = self.shape

        active_bits = np.sum(self.state)
        if active_bits > 0:
            dropout = 1 - self.max_activity/active_bits
        else:
            dropout = -1
            
        if dropout > 0:
            self.state = (self.state*np.random.sample(n**2) > dropout).astype(float)

        self.add(X)

        return self.layer


    def decode(self, radius=10):
        m = self.shape[2]
        r = radius
        feature_map = np.zeros((2*r + 1, 2*r + 1, m))
        for x in range(-r,r + 1):
            for y in range(-r ,r + 1):
                prediction  = self.explore(np.array([x,y]), mentally=True)
                feature_map[x + r, y + r] = np.sum(prediction, axis=0)

        return feature_map

    def decode_bw(self, radius=10):
        r = radius
        feature_map = self.decode(radius)
        entropy_map = np.zeros((2*r + 1, 2*r + 1))
        for x in range(-r,r + 1):
            for y in range(-r ,r + 1):
                prediction  = self.explore(np.array([x,y]), mentally=True)
                counts = feature_map[x + r, y + r]
                prob   = np.exp(counts)
                prob  /= np.sum(prob)
                entropy_map[x + r, y + r] = - entropy(prob, base=2)

        return entropy_map





