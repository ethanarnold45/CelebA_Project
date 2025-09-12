# Reminder of the paths to the numPy files of image data:
# '/Users/ethanarnold/Summer Project 25/CelebA/numPy/CelebA_testfile.npy'
# '/Users/ethanarnold/Summer Project 25/CelebA/numPy/reduced3.npy'

from ManifoldDiffusionGeometry import Scalar_Curvature
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Make coloured plot of scalar curvature
#################################################################
def ETHAN_plot_scalar_3d(data, S):
    # data is 3D data to plot
    # S is the estimates of curvature at each point which we use to plot the colours
    
    max_curvature = np.absolute(S).max()        # Used for centering the colour scale so zero curvature is white

    # Creating axis
    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]

    # Creating graph
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(X, Y, Z, c=S, cmap='coolwarm', vmin=-max_curvature, vmax=max_curvature)

    # Formatting colour bar
    #cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=30, ticks=[0])
    #cbar.set_label("Scalar Curvature", fontsize=16)
    #cbar.ax.text(0, -max_curvature-200, round(-max_curvature), ha="center", va="center", fontsize=10)
    #cbar.ax.text(0, +max_curvature+200, round(max_curvature), ha="center", va="center", fontsize=10)

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
#################################################################


celeba_reduced_to_three_dims = np.load('/Users/ethanarnold/Summer Project 25/CelebA/numPy/reduced3.npy')
celeba_reduced_to_five_dims = np.load('/Users/ethanarnold/Summer Project 25/CelebA/numPy/reduced5.npy')

S = Scalar_Curvature(celeba_reduced_to_five_dims, d=3)

ETHAN_plot_scalar_3d(celeba_reduced_to_three_dims, S)