import os
import numpy as np
import dippykit as dip
import cv2
import EnhancementLib as enh
from PIL import Image

x_max = 0
y_max = 0

dir_no = "../original_data/no/"
dir_yes = "../original_data/yes/"
dirs = [dir_no, dir_yes]

###########################
# IMAGE SIZE DISTRIBUTION #
###########################

for directory in dirs:
    for filename in os.listdir(directory):
        im = dip.im_read(directory + filename)
        dim = im.shape
        x_max = max(x_max, dim[1])
        y_max = max(y_max, dim[0])

    image_size_distribution = np.zeros((y_max + 100, x_max + 100))

    for filename in os.listdir(directory):
        im = dip.im_read(directory + filename)
        dim = im.shape
        image_size_distribution[dim[0]][dim[1]] += 1

image_size_distribution = image_size_distribution / np.amax(image_size_distribution)

dip.contour(dip.float_to_im(image_size_distribution, 8))
dip.grid()
dip.xlabel('image width')
dip.ylabel('image height')
dip.title('Image size distribution')

###########################
# CONVERTING TO GRAYSCALE #
###########################
for directory in dirs:
    for filename in os.listdir(directory):
        out = Image.open(directory+filename).convert('L')
        filename_without_extension = os.path.splitext(filename)[0]
        if directory == dir_no:
            out.save("../grayscale_data/no/" + filename_without_extension + ".png")
        elif directory == dir_yes:
            out.save("../grayscale_data/yes/" + filename_without_extension + ".png")

###########################
# RESIZING IMAGES         #
###########################
dir_no = "../grayscale_data/no/"
dir_yes = "../grayscale_data/yes/"
dirs = [dir_no, dir_yes]

for directory in dirs:
    for filename in os.listdir(directory):
        im = dip.im_read(directory + filename)
        out = dip.resize(im, (350, 300), interpolation=cv2.INTER_CUBIC)
        filename_without_extension = os.path.splitext(filename)[0]
        if directory == dir_no:
            dip.im_write(out, "../resized_data/no/" + filename_without_extension + ".png", quality=95)  # 95 is the best possible image quality
        elif directory == dir_yes:
            dip.im_write(out, "../resized_data/yes/" + filename_without_extension  + ".png", quality=95)  # 95 is the best possible image quality

###########################
# VARYING CONTRAST        #
###########################
dir_no = "../resized_data/no/"
dir_yes = "../resized_data/yes/"
dirs = [dir_no, dir_yes]

for directory in dirs:
    for filename in os.listdir(directory):
        im = dip.im_read(directory + filename)
        out = enh.add_contrast(im)
        filename_without_extension = os.path.splitext(filename)[0]
        if directory == dir_no:
            dip.im_write(out, "../contrast_data/no/" + filename_without_extension + ".png", quality=95)  # 95 is the best possible image quality
        elif directory == dir_yes:
            dip.im_write(out, "../contrast_data/yes/" + filename_without_extension  + ".png", quality=95)  # 95 is the best possible image quality