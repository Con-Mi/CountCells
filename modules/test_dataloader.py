from dataloader import CellDataLoader, CellData, CellTrainData
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import ImageFilter, Image


#dataset = CellDataLoader()
data = CellData( pd.read_csv("../data/AugmData/images.csv"),
                pd.read_csv("../data/AugmData/labels.csv"))
#data = CellData( pd.read_csv("../data/GenData/train_input_ids.csv"),
#                 pd.read_csv("../data/GenData/train_labels_ids.csv"))
#data = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

#data = CellTrainData()
#loader = torch.utils.data.DataLoader(data, shuffle=True)

for sample in tqdm(data, total=len(data)):
    img, label = sample
    #img, label = toTensor(img), toTensor(label)
    #new_img, new_label = filters.gaussian(img), filters.gaussian(label)
    new_img, new_label = img.filter(ImageFilter.GaussianBlur), label.filter(ImageFilter.GaussianBlur)
    #new_img.save("../data/AugmData/image/noise.png")
    #new_label.save("../data/AugmData/labels/noise.png")
