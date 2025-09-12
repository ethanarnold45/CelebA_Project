import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# Creates diffusion the first n diffusion maps and returns them as numPy array
#################################################################
def diffusion_coords(X, epsilon, n_components, t):
    # X is the data you are reducing given as a numpy array
    # epsilon is the epsilon value you wish to use when creating the kernels
    # n_components is the number of diffusion maps you wish to return
    # t is the number of discrete time steps you wish to take on the graph we create

    #kernel matrix
    K = rbf_kernel(X, gamma=1/(epsilon))

    #vector of degrees (inverted)
    VD = np.sum(K, axis=1)**(-1)

    #matrix of degrees (inverted)
    D = np.diag(VD)

    #transition matrix
    P = np.matmul(D,K)
    
    #evals and evects
    evals, evects = np.linalg.eig(P.transpose())

    #reordering evals and evects
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evects = evects[:,idx]

    #truncating evals and evectors
    evals = evals[0:n_components]
    evects = evects[:, 0:n_components]

    #creating and adding the diffusion maps for each data point
    diffusion_coords = np.empty((0,n_components))
    eval_matrix_power_of_t = np.linalg.matrix_power(np.diag(evals), t)
    
    for x in range(len(X)):
        coords1 = evects[x:x+1,:]
        coords = np.matmul(coords1, eval_matrix_power_of_t)
        diffusion_coords = np.append(diffusion_coords, coords, axis=0)

    return diffusion_coords     # Each diiffusion map is given as the rows of the final matrix
#################################################################