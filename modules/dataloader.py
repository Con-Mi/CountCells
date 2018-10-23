from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import skimage


class CellData(Dataset):
    def __init__(self, file_data_idx, file_label_idx, transform=None, mode="train"):
        self.data_root = "../data/GenData/"
        self.file_data_idx = file_data_idx
        self.file_label_idx = file_label_idx
        self.transform = transform
        self.mode = mode
        if self.mode is "train":
            self.data_dir = os.path.join(self.data_root, "TrainData/images/")
            self.label_dir = os.path.join(self.data_root, "TrainData/labels/")
        elif self.mode is "validation":
            pass
        elif self.mode is "test":
            pass

    def __len__(self):
        return len(self.file_data_idx)

    def __getitem__(self, index):
        #if index not in range(len(self.file_data_idx)):
        #    print("ERROR")
        #    return self.__getitem__(np.random.randint(0, self.__len__()))
        file_id = self.file_data_idx["ids"].iloc[index]
        if self.mode is "train":
            train_id = self.file_data_idx["ids"].iloc[index]
            label_id = self.file_data_idx["ids"].iloc[index]
            self.image_path = os.path.join(self.data_dir, train_id)
            self.label_path = os.path.join(self.label_dir, label_id)
            #image = Image.open(self.image_path)
            #label = Image.open(self.label_path)
            image = skimage.io.imread(self.image_path)
            label = skimage.io.imread(self.label_path)
            if self.transform is not None:
                image = self.transform(image)
                label = self.transform(image)
            return image, label
        if self.mode is "validation":
            pass
        if self.mode is "test":
            pass

def CellTrainData(data_transform=None, mode="train"):
    file_idxs = pd.read_csv("../data/GenData/train_input_ids.csv")
    label_idxs = pd.read_csv("../data/GenData/train_labels_ids.csv")
    dataset = CellData(file_idxs, label_idxs, transform=data_transform, mode=mode)
    return dataset

def CellDataLoader(data_transform=None, mode="train", batch_sz=2, workers=1):
    file_idxs = pd.read_csv("../data/GenData/train_input_ids.csv")
    label_idxs = pd.read_csv("../data/GenData/train_labels_ids.csv")
    if data_transform is None:
        data_transform = transforms.ToTensor()
    dataset = CellData(file_idxs, label_idxs, transform=data_transform, mode="train")
    dataloader = DataLoader(dataset, batch_size=batch_sz, num_workers=workers, shuffle=True)
    return dataloader
