import torch
from dense_linknet_model import denseLinkModel


inputs = torch.randn(1, 4, 512, 512)
model = denseLinkModel(input_channels=4, pretrained=True)
out = model(inputs)
print(out.size())