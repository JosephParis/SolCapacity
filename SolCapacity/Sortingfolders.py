import os
from os import listdir
from os.path import isdir, isfile, join

# ign masks =7685, images =17325, total =25010
# google masks =28807, images =13303, total =42110
base_path = "Solar_pictures/bdappv/bdappv/ign"
#google_image_path = '../Solar_pictures/bdappv/bdappv/google/img'
ign_image_path = base_path + '/img'
#google_mask_path = '../Solar_pictures/bdappv/bdappv/google/mask'
<<<<<<< HEAD
ign_mask_path= base_path + '/mask'
=======
ign_mask_path = base_path + '/mask'
>>>>>>> tmp
#base_path = "../Solar_pictures/bdappv/bdappv/ign"
files = listdir(base_path)
only_directories = [path for path in files if isdir(join(base_path,path))]
here = 0
here_again = 0
# list types ign_img, ign_masks, ign_matches, google_images, google_masks, google_matches, 
# all_images, all_masks,all_matches
ign_img = []
ign_mask = []
ign_matches = []
#images added to list
for file_path in listdir(ign_image_path):
    ign_img.append(file_path)
    full_file_path = join(ign_image_path, file_path)
#masks added to list
for file_path in listdir(ign_mask_path):
    ign_mask.append(file_path)
    full_file_path = join(ign_mask_path, file_path)
#check mask and image lists. adds to both list
for file in ign_mask:
    if file in ign_img:
        ign_matches.append(join(ign_mask_path, file_path))
#print('here' + str(here))
#print('here_again' + str(here_again))
#print('masks:' + str(len(ign_mask)))
#print('images:' + str(len(ign_img)))
#print('both:' + str(len(ign_matches)))

#onlyfiles = [f for f in os.listdir(ign_image_path) if os.path.isfile(os.path.join(ign_mask_path, f))]
