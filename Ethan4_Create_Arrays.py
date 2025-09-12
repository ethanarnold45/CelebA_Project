from Ethan3_Reducing_Dims import diffusion_coords
from PIL import Image
import numpy as np


# Changing CelebA images to vectors
##################################################################
import os
def list_vectors():
    img_folder = '/Users/ethanarnold/Summer Project 25/CelebA/Test File'        # Path to greyscaled images
    imgs = sorted(os.listdir(img_folder))
    vectors = []

    for i in imgs:

        with Image.open(os.path.join(img_folder,i)) as opened_img:
            new_vect = np.asarray(opened_img).flatten()
            vectors.append(new_vect)

    vectors = np.stack(vectors)
    return vectors
#################################################################

celeba = list_vectors() # Creates a matrix containing all the images in vector form. Each image is a row.

reduced_to_3 = diffusion_coords(celeba, 76000000, 3, 1)


# Save the vectors as .npy files to given path for faster loading in future
np.save('/Users/ethanarnold/Summer Project 25/CelebA/celeba_vectors', celeba)
np.save('/Users/ethanarnold/Summer Project 25/CelebA/reduced3', reduced_to_3)

# We also save reduced4, reduced5, etc, where celeba has been reduced to dimension 4, 5 and so on