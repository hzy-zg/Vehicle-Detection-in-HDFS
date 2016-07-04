from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.externals import joblib
import glob
import os

positive_path = "./dataset/positive/"
negative_path = "./dataset/negative/"

orientations = 9
pixels_per_cell = [8, 8]
cells_per_block = [3, 3]
visualize = False
normalize = True

features_positive = "./features/positive"
features_negative = "./features/negative"

print "Calculating positive descriptors"
positive_data =  glob.glob(os.path.join(positive_path, "*"))
negative_data =  glob.glob(os.path.join(negative_path, "*"))

count =0

for im_path in positive_data:
    im = imread(im_path, as_grey=True)
    im = resize(im, (128,128), mode='edge')

    fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
    if count == 0:
        print fd
        count+=1
    
    fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
    fd_path = os.path.join(features_positive, fd_name)
    joblib.dump(fd, fd_path)
print "Positive features Generated"

print "Calculating negative descriptors"
for im_path in negative_data:
    im = imread(im_path, as_grey=True)
    im = resize(im, (128,128), mode='edge')

    fd = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize, normalize)
    fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
    fd_path = os.path.join(features_negative, fd_name)
    joblib.dump(fd, fd_path)
print "Negative features Generated"

print "Calculated"


