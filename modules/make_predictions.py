import numpy as np
from matplotlib import pyplot as plt
from torch import sigmoid
from torchvision import transforms

import time
from PIL import Image

from helper import load_model
from dense_linknet_model import denseLinkModel
from get_data_ids import get_ids_in_list


start_time = time.time()

segm_model = denseLinkModel(input_channels=4)
segm_model = load_model(segm_model, model_dir="./dense_linknet_6.pt")
trf = transforms.Compose([ transforms.Resize(size=(384, 384)), transforms.ToTensor() ])

data_path = "../test/images/"
prediction_path = "../test/predictions/"
images = get_ids_in_list(data_path)

thrs=0.56
upper = 1
lower = 0

for img_id in images:
    img = Image.open(data_path + img_id).convert("RGBA")
    img_in = trf(img)
    img_in = img_in.unsqueeze(dim=0)
    output = segm_model(img_in)
    pred = sigmoid(output)
    pred = pred.squeeze()
    #print(pred.size())
    #pred = pred.permute(1, 2, 0)
    #pred = pred.squeeze()
    output_np = pred.detach().numpy()
    plt.imshow(output_np)
    plt.title(img_id)
    plt.show()