from dataloader import CellDataLoader, CellData, CellTrainData, CellTrainValidLoader
from torchvision import transforms
from tqdm import tqdm


mul_transf = [ transforms.Resize(size=(512, 512)), transforms.ToTensor() ]
data = CellDataLoader(data_transform=transforms.Compose(mul_transf))
tr_loader, valid_loader = CellTrainValidLoader(data_transform=transforms.Compose(mul_transf))
print("Testing the validation loader..")

for img, label in tqdm(valid_loader, total=len(valid_loader)):
    img = 0