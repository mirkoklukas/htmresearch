import numpy as np
from scipy.stats import entropy




def create_module_shape(num_modules,n, rmin=-4, rmax=4):
    shapes  = np.zeros((num_modules,2))
    for i in range(num_modules):
        shapes[i,0] = n + np.random.randint(rmin,rmax + 1)
        shapes[i,1] = n + np.random.randint(rmin,rmax + 1)

    return shapes.astype(int)


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