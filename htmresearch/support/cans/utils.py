import numpy as np
from scipy import signal



def mexican_hat(x, sigma=1.):
    a = 2./ ( np.sqrt(3*sigma) * np.power(np.pi,0.25 ) )
    b = (1. - (x/sigma)**2 )
    c = np.exp( - x**2/(2.*sigma**2))
    return a*b*c


def W_zero(x):
    a          = 1.0
    lambda_net = 4.0
    beta       = 3.0 / lambda_net**2
    gamma      = 1.05 * beta
    
    x_length_squared = x**2
    
    return a*np.exp(-gamma*x_length_squared) - np.exp(-beta*x_length_squared)



def create_W(J, D, s=1.0):
    n = D.shape[0]
    W = np.zeros(D.shape)
    W = J(s*D) 

    np.fill_diagonal(W, 0.0)
    
    for i in range(n):
        W[i,:] -= np.mean(W[i,:])
    
    return W 


def normalize(x):
    x_   = x - np.amin(x)
    amax = np.amax(x_)

    if amax != 0.:
        x_ = x_/amax
    
    return x_


def optical_flow(I1g, I2g, window_size):
    
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])
    w = window_size/2 

    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) +\
         signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)


    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            A = np.zeros((2,2))
            b = np.zeros((2,1))
            A[0,0] = np.sum(Ix**2)
            A[0,1] = np.sum(Iy*Ix)
            A[1,0] = np.sum(Iy*Ix)
            A[1,1] = np.sum(Iy**2)
            b[0,0] = - np.sum(Ix*It)
            b[1,0] = - np.sum(Iy*It)
            A_ = np.linalg.pinv(A)
            nu = np.dot(A_, np.dot(A.T,b) )
            u[i,j]=nu[0]
            v[i,j]=nu[1]
 

    return (u,v)


def mean_flow(U, t, w=200):
    return np.mean(U[t:t+w], axis=0)

def flow_to_color(u,v):
    n = len(u)
    sin_angle = np.sin(np.angle(u + v*1j)) + np.cos(np.angle(u + v*1j))
    cos_angle = np.sin(np.angle(u + v*1j)) + np.cos(np.angle(u + v*1j))

    c = np.zeros((n, 3))
    c[:,0] = (np.cos(np.angle(u + v*1j))  + 1.)/2.
    c[:,1] = (np.sin(np.angle(u + v*1j))  + 1.)/2.
    return c




def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask



