import numpy as np
from matplotlib import pyplot as plt
from torch import sigmoid
from torchvision import transforms
from torchvision import utils
from torch import nn

import time
from PIL import Image
from tqdm import tqdm

from helper import load_model
from var_dense_linknet_model import denseLinkModel
from get_data_ids import get_ids_in_list


start_time=time.time()

segm_model=denseLinkModel(input_channels=3)
segm_model=nn.DataParallel(segm_model)
segm_model=load_model(segm_model, model_dir="./var_dense_linknet_384_green_sgd_bce.pt")

img_size=384
trf = transforms.Compose([ transforms.Resize(size=(img_size, img_size)), transforms.ToTensor() ])

data_path = "../test/images/"
prediction_path = "../test/gr_predictions_bce/"
images = get_ids_in_list(data_path)

thrs=0.56
upper=1
lower=0

for img_id in tqdm(images, total=len(images)):
    img1 = Image.open(data_path + img_id)
    img1 = img1.resize(size=(img_size, img_size))
    img = Image.open(data_path + img_id).convert("RGB")
    img_in = trf(img)
    img1.save(prediction_path+img_id)
    img_in = img_in.unsqueeze(dim=0)
    output = segm_model(img_in)
    pred = sigmoid(output)
    pred = pred.squeeze()
    output_np = pred.detach().numpy()
    binary_out = np.where(output_np > thrs, upper, lower)
    #binary_out = output_np
    #mask = Image.fromarray(binary_out)
    #mask.save(img_id + "_mask.png")
    plt.imsave(prediction_path + img_id + "_mask.png", binary_out, cmap = "tab20b")
    #plt.imshow(binary_out)
    #plt.title(img_id)
    #plt.show()
