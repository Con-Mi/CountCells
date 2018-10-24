import os
import skimage
import sys
from tqdm import tqdm
import numpy as np
from PIL import Image


TRAIN_PATH = "../data/DSB-Stage1/"
#TEST_PATH = "../"

train_ids = next(os.walk(TRAIN_PATH))[1]
traind_ids = sorted(train_ids)
print("Getting images and reconstructing masks..")
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH+id_
    img = skimage.io.imread(path + "/images/" + id_ + ".png")
    #skimage.io.imsave("../data/GenData/TrainData/images/" + str(n) + "_.png", img)
    im = Image.fromarray(img)
    im.save("../data/GenData/TrainData/images/" + str("%04d" % n) + "_.png")
    
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
    for mask_file in next(os.walk(path+"/masks/"))[2]:
        mask_ = skimage.io.imread(path + "/masks/" + mask_file)
        mask = np.maximum(mask, mask_)
    #skimage.io.imsave("../data/GenData/TrainData/labels/" + str(n) + "_.png", mask)
    mask_im = Image.fromarray(mask)
    mask_im.save("../data/GenData/TrainData/labels/" + str("%04d" % n) + "_.png")
    