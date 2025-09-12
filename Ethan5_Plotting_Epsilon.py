# Reminder of the paths to the numPy files of image data:
# '/Users/ethanarnold/Summer Project 25/CelebA/numPy/CelebA_testfile.npy'

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

X = np.load('/Users/ethanarnold/Summer Project 25/CelebA/numPy/CelebA_testfile.npy')

def sum_weights(epsilon):
    K = rbf_kernel(X, gamma=1/(epsilon))
    sum = np.sum(K)
    return sum

eps = np.logspace(np.log10(10000000), np.log10(500000000), num=50, endpoint=True, base=10.0, dtype=None, axis=0)
sums = np.array([sum_weights(e) for e in eps])

fig, ax = plt.subplots(constrained_layout=True)
ax.loglog(eps, sums, 'x-k', linewidth=1, color='black', markersize=4)

ax.axvline(x=7.6e+07, color ='black', linestyle='--', linewidth=1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel(r"$\epsilon$", fontsize=25)
ax.set_ylabel(r"$\sum_{i,j} \, W_{ij}(\epsilon)$", rotation=0, fontsize=25)
ax.yaxis.set_label_coords(-0.25, 0.35)
ax.tick_params(axis='both', which='major', labelsize=17)

plt.show()
