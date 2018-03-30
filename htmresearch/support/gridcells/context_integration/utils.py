import numpy as np
from scipy.stats import entropy




def create_module_shape(num_modules,n, rmin=-6, rmax=6):
    shapes  = np.zeros((num_modules,2))
    for i in range(num_modules):
        shapes[i,0] = n + np.random.randint(rmin,rmax + 1)
        shapes[i,1] = n + np.random.randint(rmin,rmax + 1)

    return shapes.astype(int)


def get_3d_actions(m):
    dx   = np.zeros(2*m)
    dy   = np.zeros(2*m)
    dz   = np.zeros(2*m)

    dx[0:3] = np.array([1,0,0])
    dy[0:3] = np.array([0,1,0])
    dz[0:3] = np.array([0,0,1])
    return dx,dy,dz

def create_action_tensor(m):
    action_tensor = np.zeros((m, 2, 2*m))
    for i in range(m):
        theta = np.random.sample()*np.pi*2
        theta2 = np.random.sample()*np.pi*2
        theta3 = np.random.sample()*np.pi*2
        sixty = np.pi/3 + np.random.randn()*0.0
        phi   = np.random.sample()*np.pi*2
        s1     = 3 + np.random.sample()*3
        s2     = 3 + np.random.sample()*3.
        s3     = 3 + np.random.sample()*3.
        
        action_tensor[i,:,0:3] = np.array([
            [ s1*np.cos(theta), s2*np.cos(theta2), s3*np.cos(theta3)],
            [ s1*np.sin(theta), s2*np.sin(theta2), s3*np.cos(theta3)]  
        ]).astype(int)

    return action_tensor

def create_env_nbh_tensor(environment, radius):
    env = environment
    e0   = env.shape[0]
    e1   = env.shape[1]
    r   = radius
    env_tensor = np.zeros((e0,e1,2*r + 1, 2*r + 1))
    for x in range(env.shape[0]):
        for y in range(env.shape[1]):
            xs = [  k%e0 for k in range(x - r ,x + r + 1 )]
            ys = [  k%e1 for k in range(y - r ,y + r + 1 )] 
            env_snip = env[xs,:][:,ys]
            env_tensor[x,y,:,:] = env_snip[:,:]

    return env_tensor


def position_estimate(env_tensor,context, radius):

    assert env_tensor.shape[2] == context.shape[0]
    assert env_tensor.shape[3] == context.shape[1]

    context = context.reshape((1,1,context.shape[0], context.shape[1]))
    
    heat = np.sum(env_tensor*context, axis=(2,3))
    prob = np.exp(heat)
    # prob = heat
    prob = prob/np.sum(prob)
    return prob


def diffusion(arr):
    arr_ = arr.copy()
    arr_+=0.2*np.roll(arr,shift=1,axis=1) # right
    arr_+=0.2*np.roll(arr,shift=-1,axis=1) # left
    arr_+=0.2*np.roll(arr,shift=1,axis=0) # down
    arr_+=0.2*np.roll(arr,shift=-1,axis=0) # up
    
    arr_+=0.1*np.roll(arr,shift=(-1,1),axis=(0,1)) 
    arr_+=0.1*np.roll(arr,shift=(-1,-1),axis=(0,1)) 
    arr_+=0.1*np.roll(arr,shift=(1,1),axis=(0,1)) 
    arr_+=0.1*np.roll(arr,shift=(1,-1),axis=(0,1)) 
    
    return arr_



def get_closed_3d_path(num_samples=20, radius=10):
    R = radius
    X = np.zeros((num_samples+1,3))
    V = np.zeros((num_samples,3))

    for i in range(num_samples):
        x = np.random.randint(-R,R, size=(3))
        X[i] = x[:]

    for i in range(num_samples):
        V[i] = X[(i+1)%num_samples] - X[i]

    X[-1] = X[0]
    return X, V




def encode(digit, l, w=4):
    start = digit * l/10
    end = start + w
    return [ i%l for i in range(start, end)]


def load_digit_features(w, shape):
    d,l = shape
    F = np.zeros((10,d,l))
    for i in range(10):
        F[i,:,encode(i%10, l, w)] = 1

    return F
    





