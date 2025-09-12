import os
import shutil

og_file = '/Users/ethanarnold/Summer Project 25/CelebA/Greyscaled Images'   # Path to folder greyscaled images 
test_file = '/Users/ethanarnold/Summer Project 25/CelebA/Test File'         # Path to where you want subset to be saved
imgs = sorted(os.listdir(og_file))

img_paths = []
new_paths = []
for i in imgs[0:2000]:                                                      # This is where you decide what images you want to save (first 2000 in this case)
    img_paths.append(os.path.join(og_file, i))
    new_paths.append(os.path.join(test_file,i))


def main():
    for old, new in zip(img_paths, new_paths):
        shutil.copyfile(old,new)