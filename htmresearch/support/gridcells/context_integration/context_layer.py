import numpy as np
from scipy.stats import entropy






class ContextLayer(object):
    """
    An experimental L6 implementation that accumulates sensory inputs and whose
    activity encodes spatial context of the inputs....
    """
    def __init__(self, layer_height, module_shapes, action_map, max_activity=10000):
        """
        Args
        ----
            layer_height: 
                The number of cells in a "minicolumn" of the layer.
            module_shapes: 
                An array of shape (m,2) describing the dimensions 
                of m grid cell modules.
            action_map: 
                An array of shape (m,2,2*m) where the i'th entry action_map[i]
                is a matrix translating 2m-dimensional motorcommand into 2-dimensional
                positional updates that can be consumed by the i'th grid module.
            max_activity: 
                The maximal number of active bits in the layer. If the number of active bits
                exceeds this bound random dropout will be applied.
        """
        m  = len(module_shapes)
        gc =  np.sum([ np.prod(module_shapes[i])  for i in range(m)  ])
        d  = layer_height
        l  = gc//d + gc%d 
        c  = d*l
        self.layer_shape   = (d, l)
        self.num_grid_cells= gc
        self.num_cells     = c
        self.module_shapes = module_shapes
        self.num_modules   = m
        self.module_bounds = np.zeros(m+1).astype(int)
        for i in range(1,m+1):
            self.module_bounds[i] = self.module_bounds[i-1] + np.prod(module_shapes[i-1])



        self.max_activity = max_activity
        self.action_map   = action_map
        
        self.perm     = np.random.permutation(c)
        self.perm_inv = np.zeros(c).astype(int)
        for i in range(c):
            self.perm_inv[self.perm[i]] = i

        self.state        = np.zeros(d*l)
        self.state_counts = np.zeros(d*l)

    def clear(self):
        self.state[:] = 0

    def get_random_anchor(self):
        m = self.num_modules
        A = np.zeros(self.num_cells)
        perm_inv = self.perm_inv
        for i in range(1,m+1):
            b1 = self.module_bounds[i-1]
            b2 = self.module_bounds[i]
            r = np.random.randint(b1, b2)
            A[r] = 1.0

        return A[perm_inv].reshape(self.layer_shape)


    def get_module(self, i):
        perm    = self.perm
        b1      = self.module_bounds[i]
        b2      = self.module_bounds[i+1]
        m_shape = self.module_shapes[i]
        m_state = self.state[perm]
        return m_state[b1:b2].reshape(m_shape)

    def highlight_module(self, i):
        perm     = self.perm
        perm_inv = self.perm_inv
        state = np.zeros(self.state.shape)
        b1 = self.module_bounds[i]
        b2 = self.module_bounds[i+1]
        m_state = self.state[perm]
        m_state[b1:b2] = 1.
        return m_state[perm_inv].reshape(self.layer_shape)

    def highlight_unused_states(self):
        perm     = self.perm
        perm_inv = self.perm_inv
        m  = self.num_modules
        bm = self.module_bounds[m]
        m_state = self.state[perm]
        m_state[bm:] = 1
        return m_state[perm_inv].reshape(self.layer_shape)

    @property
    def layer(self):
        return self.state.reshape(self.layer_shape)


    @property
    def layer_of_counts(self):
        return self.state_counts.reshape(self.layer_shape)


    def _explore(self, A, mentally=False):

        perm     = self.perm  
        perm_inv = self.perm_inv            
        m        = self.num_modules
        m_state  = self.state[perm]
        m_state_counts = self.state[perm]

        for i in range(m):
            n1, n2 = self.module_shapes[i]
            b1     = self.module_bounds[i]
            b2     = self.module_bounds[i+1]
            C      = self.get_module(i)
            C_     = np.zeros((n1,n2))
            
            for x0 in range(n1):
                for x1 in range(n2):
                    y0 =(x0 + A[i,0])%n1
                    y1 =(x1 + A[i,1])%n2

                    C_[y0, y1] += C[x0, x1]
        
            C_ = np.clip(C_, 0, 1)
            m_state[b1:b2] = C_.reshape(-1)[:]


        if mentally == False:
            self.state = m_state[perm_inv] 
            return self.layer
        else:
            return m_state[perm_inv].reshape(self.layer_shape)

    def _explore2(self, A, mentally=False):

        perm     = self.perm  
        perm_inv = self.perm_inv            
        m        = self.num_modules
        m_state  = self.state_counts[perm]

        for i in range(m):
            n1, n2 = self.module_shapes[i]
            b1     = self.module_bounds[i]
            b2     = self.module_bounds[i+1]
            C      = self.get_module(i)
            C_     = np.zeros((n1,n2))
            
            for x0 in range(n1):
                for x1 in range(n2):
                    y0 =(x0 + A[i,0])%n1
                    y1 =(x1 + A[i,1])%n2

                    C_[y0, y1] = C[x0, x1]
        
            m_state[b1:b2] = C_.reshape(-1)[:]


        if mentally == False:
            self.state = m_state[perm_inv] 
            return self.layer
        else:
            return m_state[perm_inv].reshape(self.layer_shape)


    def explore(self, a, mentally=False):
        m = self.num_modules
        A = np.zeros((m, 2)).astype(int)
        for i in range(m):
            A[i] = np.dot(self.action_map[i], a)

        return self._explore(A, mentally)

    def explore2(self, a, mentally=False):
        m = self.num_modules
        A = np.zeros((m, 2)).astype(int)
        for i in range(m):
            A[i] = np.dot(self.action_map[i], a)

        return self._explore2(A, mentally)

    def add(self, X):
        """
        Extend the current context 
        (or state respectively)
        """
        assert X.shape == self.layer_shape

        self.state += X.reshape(-1)
        self.state  = np.clip(self.state, 0,1)

        return self.layer

    def add_to_counts(self, X):
        """
        Extend the current context 
        (or state respectively)
        """
        assert X.shape == self.layer_shape

        self.state_counts += X.reshape(-1)

        return self.layer_of_counts


    def extend(self, a, X):
        """Extend the current context"""
        m = self.num_modules
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
            self.state = (self.state*np.random.sample(self.num_cells) > dropout).astype(float)

        self.add(X)
        
        return self.layer

    def feed_bottom_up(self, column_indices):
        d, l = self.layer_shape 
        X = np.zeros((d,l))
        X[:,column_indices] = 1.0
        self.add(X) 

    def decode(self, radius=10):
        l = self.layer_shape[1]
        m = self.num_modules
        r = radius

        feature_map = np.zeros((2*r + 1, 2*r + 1, l))

        for x in range(-r,r + 1):
            for y in range(-r ,r + 1):
                v = np.zeros(2*m) 
                v[0] = x
                v[1] = y
                prediction  = self.explore(v, mentally=True)
                feature_map[x + r, y + r] = np.sum(prediction, axis=0)

        return feature_map


    def decode_bw(self, radius=10, normalize=True, threshold=None, softmax=False):
        r = radius
        feature_map = self.decode(radius)
        entropy_map = np.zeros((2*r + 1, 2*r + 1))
        for x in range(-r,r + 1):
            for y in range(-r ,r + 1):
                counts = feature_map[x + r, y + r]
                if softmax==True:
                    prob   = np.exp(counts)
                else:
                    prob = counts
                prob  /= np.sum(prob)
                entropy_map[x + r, y + r] = - entropy(prob, base=2)

        if normalize==True:
            entropy_map -= np.amin(entropy_map)
            if np.amax(entropy_map) > 0:
                entropy_map /= np.amax(entropy_map)
                
        if threshold != None:
            entropy_map = (entropy_map >= threshold).astype(float)

        return entropy_map


    def __str__(self):
        summary = "\n**Context Layer:**"\
                  "\n------------------"\
                  "\nNumber of cells:\t {self.num_cells}"\
                  "\nLayer Shape:\t\t {self.layer_shape}"\
                  "\nHyper-Module Shapes:\n{self.module_shapes}"\
                  "\nModule bounds: {self.module_bounds}"\
                  "\nNumber of grid cells:\t {self.num_grid_cells}"\
                  "\nActivity bound:\t\t {self.max_activity}"\
                  "\n------------------".format(self=self)
                  
        return summary


