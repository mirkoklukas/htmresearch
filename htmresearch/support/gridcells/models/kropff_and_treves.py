"""
Implementation of the network described in

>   E. Kropff and A. Treves, "The emergence of grid cells: intelligent design or just adaption?" 
    Hippocampus (2008), 1256-69

Notation has mostly been adapted from the paper. However, 
I changed the input notation from an $r$ to an $x$, and
the hidden activations from $r_{(in)act}$ to $h_{(in)act}$.
"""
import numpy as np



def create_parameters(num_in, num_out):
    n, m = num_out, num_in
    parameters = {}
    parameters["J"]      = np.random.random((n,m))
    parameters["theta"]  = 0.
    parameters["g"]      = 1.
    parameters["epsilon"]= 0.05
    # parameters["b1"]     = 0.1
    # parameters["b2"]     = 0.033  # The paper suggests: b_2 := b_1 / 3
    parameters["b1"]     = 0.3
    parameters["b2"]     = 0.1  
    parameters["b3"]     = 0.005
    parameters["b4"]     = 0.005
    parameters["psi_sat"]= 30.
    parameters["a0"]     = 3.  # The paper suggests: a_0 := 0.1*psi_sat
    parameters["s0"]     = 0.3
    parameters["mean_y"] = np.ones(n)*0.01
    parameters["mean_x"] = np.ones(m)*0.01
    parameters["h_act"]  = np.zeros(n)
    parameters["h_inact"]= np.zeros(n)
    return parameters

def clip_parameters(parameters):
    theta    = parameters["theta"]
    g        = parameters["g"]

    theta = np.clip(theta, -10.  , 10.)
    g     = np.clip(g    ,   0.01,  2.)

    parameters["theta"] = theta
    parameters["g"]     = g

def heaviside(x, h0=0.5):
    """
    Implementation of the heaviside function.
    """
    result = np.zeros(x.shape)            
    result[np.where(x  < 0)[0]] = 0.0
    result[np.where(x == 0)[0]] = h0
    result[np.where(x >  0)[0]] = 1.0
    return result


def hidden_activation(J, x):
    """
    Computes the (1st layer) hidden activation $h_t$
    given in the paper.
    """
    n   = J.shape[1]
    x   = x.reshape((-1,1))
    res = np.dot(J, x)/n
    res = res.reshape(-1)
    return res


def psi(h, psi_sat, theta, g):
    """
    The ``transfer function'' (or activation function) 
    given in the paper.
    """
    res = psi_sat * (2/np.pi) * np.arctan(g*(h - theta)) * heaviside(h - theta)
    return res 


def moving_average(old, b, learned):
    """
    An Exponential moving average.
    """
    return (1-b)*old + b*learned


def sparseness(y):
    """
    Measure of sparseness. Chose a different 
    penalty than the one given in the paper.
    """
    penalty = np.sum(y)
    return penalty


def mean_activity(y):
    """
    Computes the mean activity, obviously.
    """
    a = np.mean(y)
    return a


def update_parameters(x, y, parameters):
    """
    Method that updates the parameters in place 
    according to the equations in the paper.
    """
    J        = parameters["J"]
    theta    = parameters["theta"]
    g        = parameters["g"]
    mean_y   = parameters["mean_y"]
    mean_x   = parameters["mean_x"]
    b3       = parameters["b3"]
    b4       = parameters["b4"]
    a0       = parameters["a0"]
    s0       = parameters["s0"]
    epsilon  = parameters["epsilon"]

    

    # Note that in the paper they don't use 
    # an exponential moving average.
    mean_y[:] = moving_average(old=mean_y, b=0.1, learned=y)
    mean_x[:] = moving_average(old=mean_x, b=0.1, learned=x)

    a = mean_activity(y)
    s = sparseness(y)
    theta = theta + b3 * (a - a0)
    g     =     g + b4*g*(s - s0)
    parameters["theta"] = theta
    parameters["g"]     = g


    dJ   =  np.dot(np.transpose([y]), [x]) 
    dJ  -=  np.dot(np.transpose([mean_y]), [mean_x]) 

    J[:,:] += epsilon * dJ

    # Normalize the synaptic weights.
    # This is NOT applied in the paper
    # norms = np.sqrt(np.sum( J**2, axis=0, keepdims=True))
    # J[:,:] = J/norms




def compute_output(x, y, parameters):
    J        = parameters["J"]
    psi_sat  = parameters["psi_sat"]
    theta    = parameters["theta"]
    g        = parameters["g"]
    b1       = parameters["b1"]
    b2       = parameters["b2"]
    h_act    = parameters["h_act"]
    h_inact  = parameters["h_inact"]
    
    h = hidden_activation(J, x)
    h_act[:]   = moving_average(old=h_act,   b=b1, learned=h - h_inact )
    h_inact[:] = moving_average(old=h_inact, b=b2, learned=h           )
        

    z = psi(h_act, psi_sat, theta, g)

    y[:] = z






    
