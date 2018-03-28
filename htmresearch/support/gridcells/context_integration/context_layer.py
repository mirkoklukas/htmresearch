import numpy as np

from scipy.stats import entropy




class ContextLayer(object):

    def __init__(self, layer_shape, module_shape, action_map, max_activity=10000):
        assert module_shape[1]       == module_shape[2],      "Check module dimensions, we want a square shaped modules..."
        assert np.prod(module_shape) == np.prod(layer_shape), "Check layer dimensions..."

        self.layer_shape  = layer_shape
        self.module_shape = module_shape
        self.max_activity = max_activity

        self.action_map = action_map

        c = np.prod(module_shape)
        self.num_cells = c
        self.perm = np.random.permutation(c)
        self.perm_inv = np.zeros(c).astype(int)
        for i in range(c):
            self.perm_inv[self.perm[i]] = i

        self.state = np.zeros(c)

    def clear(self):
        self.state[:] = 0

    @property
    def module(self):
        perm = self.perm
        return self.state[perm].reshape(self.module_shape)

    def set_individual_module(self, i, M):
        perm     = self.perm
        perm_inv = self.perm_inv
        module =  self.state[perm].reshape(self.module_shape)
        module[i] = M[:]
        self.state = module.reshape(-1)[perm_inv]


    @property
    def layer(self):
        return self.state.reshape(self.layer_shape)

    def _explore(self, A, mentally=False):

        perm_inv = self.perm_inv        
        C  = self.module
        C_ = np.zeros(self.module_shape)

        m, n, _ = self.module_shape
        for i in range(m):
            for x0 in range(n):
                for x1 in range(n):
                    y0 =(x0 + A[i,0])%n
                    y1 =(x1 + A[i,1])%n

                    C_[i, y0, y1] += C[i, x0, x1]
        
        C_ = np.clip(C_, 0, 1)

        if mentally == False:
            self.state = C_.reshape(-1)[perm_inv]
            return self.layer
        else:
            return C_.reshape(-1)[perm_inv].reshape(self.layer_shape)

    def explore(self, a, mentally=False):
        m, n, _ = self.module_shape
        A = np.zeros((m, 2)).astype(int)
        for i in range(m):
            A[i] = np.dot(self.action_map[i], a)

        return self._explore(A, mentally)

    def intersect(self, X):
        assert X.shape == self.layer_shape

        self.state *= X.reshape(-1)
        self.state  = np.clip(self.state, 0,1)

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
        m, n, _ = self.module_shape
        A = np.zeros((m, 2)).astype(int)

        for i in range(m):
            A[i] = np.dot(self.action_map[i], a)

        self._explore(A)


        active_bits = np.sum(self.state)

        if active_bits > 0:
            dropout = 1 - self.max_activity/active_bits
        else:
            dropout = -1
            
        if dropout > 0:
            self.state = (self.state*np.random.sample(m*n*n) > dropout).astype(float)

        self.add(X)

        return self.layer


    def decode(self, radius=10):
        m = self.layer_shape[1]
        num_modules = self.module_shape[0]
        r = radius
        feature_map = np.zeros((2*r + 1, 2*r + 1, m))
        for x in range(-r,r + 1):
            for y in range(-r ,r + 1):
                v = np.zeros(2*num_modules) 
                v[0] = x
                v[1] = y
                prediction  = self.explore(v, mentally=True)
                feature_map[x + r, y + r] = np.sum(prediction, axis=0)

        return feature_map


    def decode_bw(self, radius=10):
        r = radius
        feature_map = self.decode(radius)
        entropy_map = np.zeros((2*r + 1, 2*r + 1))
        for x in range(-r,r + 1):
            for y in range(-r ,r + 1):
                counts = feature_map[x + r, y + r]
                prob   = np.exp(counts)
                prob  /= np.sum(prob)
                entropy_map[x + r, y + r] = - entropy(prob, base=2)

        return entropy_map


    def __str__(self):
        summary = "**Context Layer:**"\
                  "\nNumber of cells:\t {self.num_cells}"\
                  "\nLayer Shape:\t\t {self.layer_shape}"\
                  "\nHyper-Module Shape:\t {self.module_shape}"\
                  "\nActivity bound:\t\t {self.max_activity}".format(self=self)
                  
        return summary


