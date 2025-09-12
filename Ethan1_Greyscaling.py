import os
from PIL import Image
from multiprocessing import Pool, cpu_count

img_folder = '/Users/ethanarnold/Summer Project 25/CelebA/Original Images'      # Path to coloured images
grey_folder = '/Users/ethanarnold/Summer Project 25/CelebA/Greyscaled Images'   # Path to where you want greyscaled images to go

img_names = sorted(os.listdir(img_folder))


def process_img(n):

    og_path = os.path.join(img_folder, n)
    grey_path = os.path.join(grey_folder, n)

    if os.path.exists(grey_path):
        return 
    
    try:
        with Image.open(og_path) as img:
            grey_img = img.convert("L")
            grey_img.save(grey_path)
        return 
    
    except Exception as e:
        return f"Error {n}: {e} \n"
   
    
def main():
    errors = []
    num_errors = 0
    num_workers = max(1,cpu_count() - 4)

    with Pool(processes=num_workers) as pool:

        for idx, result in enumerate(pool.imap_unordered(process_img,img_names), start=1):
            if result:
                errors.append(result)
                num_errors = num_errors + 1

            if idx % 2000 == 0 or idx == len(img_names):
                print(f"{idx} images processed. {num_errors} errors.")

        if len(errors) == 0:
            print("Complete. No errors occurred")
        
        else:
            print(f"Complete. The following {len(errors)} errors occurred:\n" + "".join(errors))