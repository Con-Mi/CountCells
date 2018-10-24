from dataloader import CellDataLoader, CellData, CellTrainData
from torchvision import transforms
from tqdm import tqdm


mul_transf = [ transforms.Resize(size=(512, 512)), transforms.ToTensor() ]
data = CellDataLoader(data_transform=transforms.Compose(mul_transf))

for img, label in tqdm(data, total=len(data)):
    img = 0
    #print(label.size())
    #print(img.size())
    #if i==2:
    #    break