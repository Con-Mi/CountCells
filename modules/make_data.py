import os
import sys

from skimage import io
from tqdm import tqdm
import numpy as np
from PIL import Image

import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
from skimage.morphology import dilation, watershed, square, erosion
from skimage.measure import label, regionprops


TRAIN_PATH = "../data/DSB-Stage1/"
#TEST_PATH = "../"

train_ids = next(os.walk(TRAIN_PATH))[1]
traind_ids = sorted(train_ids)
print("Getting images and reconstructing masks..")
sys.stdout.flush()

def create_contour(labels):
    mask = labels.copy()
    mask[mask > 0] = 1
    dilated = binary_dilation(mask, iterations=10)
    mask_w1 = watershed(dilated, labels, mask=dilated, watershed_line=True)
    mask_w1[mask_w1 > 0] = 1
    contours = dilated - mask_w1
    contours = binary_dilation(contours, iterations=1)
    return contours

# In Progress

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH+id_
    img = io.imread(path + "/images/" + id_ + ".png")
    im = Image.fromarray(img)
    im.save("../data/GenData/TrainData/images/" + str("%04d" % (n + 65)) + "_.png")
    
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
    for mask_file in next(os.walk(path+"/masks/"))[2]:
        mask_ = io.imread(path + "/masks/" + mask_file)
        mask = np.maximum(mask, mask_)
    mask_im = Image.fromarray(mask)
    mask_im.save("../data/GenData/TrainData/labels/" + str("%04d" % (n + 65)) + "_.png")
