# Reminder of the paths to the numPy files of image data:
# '/Users/ethanarnold/Summer Project 25/CelebA/numPy/CelebA_testfile.npy'
# '/Users/ethanarnold/Summer Project 25/CelebA/numPy/reduced3.npy'

from ManifoldDiffusionGeometry import Dimension_Estimate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors


# Make 3D plot from 'data' and assign a colour to each point based on 'dims'
def ETHAN_plot_dims_3d(data, dims):

    colors=["#ff0000", "#ff7300", "#e3ef00","#44ff00", "#8ab6ee", "#000dff","#ae00ff"]
    colormap = mcolors.ListedColormap(colors)

    bounds = [-0.1, 0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    mynorm = mcolors.BoundaryNorm(bounds, len(colors), clip=True)

    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z, c=dims, cmap=colormap, norm=mynorm)

    # Removing background
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    ax.set_xlabel(r"$\Psi_1$")
    ax.set_ylabel(r"$\Psi_2$")
    ax.set_zlabel(r"$\Psi_3$")

    # shadow on XY plane
    ax.scatter(X, Y, np.full_like(Z, min(Z)), c='gray', alpha=0.05, s=10)

    plt.show()


celeba_reduced_to_3_dims = np.load('/Users/ethanarnold/Summer Project 25/CelebA/numPy/reduced3.npy')
celeba_reduced_to_5_dims = np.load('/Users/ethanarnold/Summer Project 25/CelebA/numPy/reduced5.npy')

pointwise_dimensions = Dimension_Estimate(celeba_reduced_to_5_dims)[1]      # Uses Iolo's tools to estimate the local dimension at each point

ETHAN_plot_dims_3d(celeba_reduced_to_3_dims, pointwise_dimensions)