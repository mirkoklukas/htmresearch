import numpy as np


def create_env_nbh_tensor(environment, radius, num_features):
    env = environment
    e0  = env.shape[0]
    e1  = env.shape[1]
    r   = radius
    env_tensor = np.zeros((num_features, e0,e1,2*r + 1, 2*r + 1))
    for f in range(1, num_features+1):
        for x in range(env.shape[0]):
            for y in range(env.shape[1]):
                xs = [  k%e0 for k in range(x - r ,x + r + 1 )]
                ys = [  k%e1 for k in range(y - r ,y + r + 1 )] 
                env_snip = (env[xs,:][:,ys] == f).astype(int)
                env_tensor[f-1,x,y,:,:] = env_snip[:,:]

    return env_tensor


def position_estimate_2(env_tensor,context, radius):

    assert env_tensor.shape[2] == context.shape[0]
    assert env_tensor.shape[3] == context.shape[1]

    context = context.reshape((1,1,context.shape[0], context.shape[1]))
    
    overlap = np.sum(env_tensor*context, axis=(2,3)) 
    prob = (overlap == np.sum(context)).astype(float) + 0.001

    prob = prob/np.sum(prob)
    return prob

