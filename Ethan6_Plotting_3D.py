# Reminder of the paths to the numPy files of image data:
# '/Users/ethanarnold/Summer Project 25/CelebA/numPy/CelebA_testfile.npy'
# '/Users/ethanarnold/Summer Project 25/CelebA/numPy/reduced3.npy'

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm, colors
import numpy as np

coords = np.load('/Users/ethanarnold/Summer Project 25/CelebA/numPy/reduced3.npy')
X = coords[:,0]
Y = coords[:,1]
Z = coords[:,2]

numberofimages = len(Z)

#distances from a reference plane (for colouring the graph)
distances = []
for i in range(0, numberofimages):
    d = coords[i][1]
    distances.append(d)


# normalize distances to [0,1] for color mapping
norm = colors.TwoSlopeNorm(vcenter=0, vmin=min(distances), vmax=max(distances))
cmap = cm.rainbow

# turn distances into RGBA colors
point_colors = cmap(norm(distances))

# creating the graph
ax = plt.axes(projection="3d")
ax.scatter(X, Y, Z, color=point_colors)

ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)

ax.set_xlabel(r"$\Psi_1$")
ax.set_ylabel(r"$\Psi_2$")
ax.set_zlabel(r"$\Psi_3$")

# shadow on XY plane
ax.scatter(X, Y, np.full_like(Z, min(Z)), c='gray', alpha=0.05, s=10)

plt.show()
