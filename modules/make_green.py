import os
from skimage import io
from skimage import color
import sys
from tqdm import tqdm
import numpy as np
from PIL import Image

TRAIN_PATH = "../data/DSB-Stage1/"

train_ids = next(os.walk(TRAIN_PATH))[1]
traind_ids = sorted(train_ids)
print("Getting and reconstructing images and masks..")
sys.stdout.flush()
green = [0, 10, 0]

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH+id_
    img = io.imread(path + "/images/" + id_ + ".png")
    img = color.rgba2rgb(img)
    img_g = green*img
    img_g = (img_g*255).astype(np.uint8)
    im = Image.fromarray(img_g)
    im.save("../data/GenData/TrainData/images/" + str("%04d" % n) + "_.png")

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
    for mask_file in next(os.walk(path+"/masks/"))[2]:
        mask_ = io.imread(path + "/masks/" + mask_file)
        mask = np.maximum(mask, mask_)
    mask_im = Image.fromarray(mask)
    mask_im.save("../data/GenData/TrainData/labels/" + str("%04d" % n) + "_.png")