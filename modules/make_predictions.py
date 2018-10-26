import numpy as np
from matplotlib import pyplot as plt
from torch import sigmoid
from torchvision import transforms
from torchvision import utils

import time
from PIL import Image

from helper import load_model
from dense_linknet_model import denseLinkModel
from get_data_ids import get_ids_in_list


start_time = time.time()

segm_model = denseLinkModel(input_channels=4)
segm_model = load_model(segm_model, model_dir="./dense_linknet_16.pt")
trf = transforms.Compose([ transforms.Resize(size=(384, 384)), transforms.ToTensor() ])

data_path = "../test/images/"
prediction_path = "../test/predictions/"
images = get_ids_in_list(data_path)

thrs=0.56
upper = 1
lower = 0

for img_id in images:
    img1 = Image.open(data_path + img_id)
    img1 = img1.resize(size=(384, 384))
    img = Image.open(data_path + img_id).convert("RGBA")
    img_in = trf(img)
    img1.save(prediction_path+img_id)
    img_in = img_in.unsqueeze(dim=0)
    output = segm_model(img_in)
    pred = sigmoid(output)
    pred = pred.squeeze()
    output_np = pred.detach().numpy()
    binary_out = np.where(output_np > thrs, upper, lower)
    #mask = Image.fromarray(binary_out)
    #mask.save(img_id + "_mask.png")
    plt.imsave(prediction_path + img_id + "_mask.png", binary_out)
    #plt.imshow(binary_out)
    #plt.title(img_id)
    #plt.show()
