import torch 
from torch import nn
from torch import optim
from torchvision import transforms

from dense_linknet_model import denseLinkModel
from helper import jaccard, dice, save_model
from dataloader import CellDataLoader, CellTrainValidLoader

import time
import copy
from tqdm import tqdm


use_cuda = torch.cuda.is_available()
# Hyperparameters
batch_size = 2
nr_epochs = 20
momentum = 0.95
lr_rate = 0.03
milestones = [5, 7, 8, 10, 12, 14, 16, 17, 18]
img_size = 384
gamma = 0.5

segm_model = denseLinkModel(input_channels=4, pretrained=True)
if use_cuda:
    segm_model.cuda()

mul_transf = [ transforms.Resize(size=(img_size, img_size)), transforms.ToTensor() ]

optimizerSGD = optim.SGD(segm_model.parameters(), lr=lr_rate, momentum=momentum)
criterion = nn.BCEWithLogitsLoss().cuda() if use_cuda else nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.MultiStepLR(optimizerSGD, milestones=milestones, gamma=gamma)

train_loader, valid_loader = CellTrainValidLoader(data_transform=transforms.Compose(mul_transf), batch_sz=batch_size, workers=20)

dict_loaders = {"train":train_loader, "valid":valid_loader}

def train_model(cust_model, dataloaders, criterion, optimizer, num_epochs, scheduler=None):
    start_time = time.time()
    val_acc_history = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(cust_model.state_dict())

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("_"*15)
        for phase in ["train", "valid"]:
            if phase == "train":
                cust_model.train()
            if phase == "valid":
                cust_model.eval()
            running_loss = 0.0
            jaccard_acc = 0.0
            dice_loss = 0.0

            for input_img, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                input_img = input_img.cuda() if use_cuda else input_img
                labels = labels.cuda() if use_cuda else labels

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    out = cust_model(input_img)
                    preds = torch.sigmoid(out)                
                    loss = criterion(preds, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input_img.size(0)
                jaccard_acc += jaccard(labels, preds)
                #dice_acc += dice(labels, preds)
            
            epoch_loss = running_loss / len(dataloaders[phase])
            aver_jaccard = jaccard_acc / len(dataloaders[phase])
            #aver_dice = dice_acc / len(dataloaders[phase])

            print("| {} Loss: {:.4f} | Jaccard Average Acc: {:.4f} |".format(phase, epoch_loss, aver_jaccard))
            print("_"*15)
            if phase == "valid" and aver_jaccard > best_acc:
                best_acc = aver_jaccard
                best_model_wts = copy.deepcopy(cust_model.state_dict)
            if phase == "valid":
                val_acc_history.append(aver_jaccard)
        print("^"*15)
        print(" ")
        scheduler.step()
    time_elapsed = time.time() - start_time
    print("Training Complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best Validation Accuracy: {:.4f}".format(best_acc))
    best_model_wts = copy.deepcopy(cust_model.state_dict())
    cust_model.load_state_dict(best_model_wts)
    return cust_model, val_acc_history

segm_model, acc = train_model(segm_model, dict_loaders, criterion, optimizerSGD, nr_epochs, scheduler=scheduler)
save_model(segm_model, name="dense_linknet_20.pt")