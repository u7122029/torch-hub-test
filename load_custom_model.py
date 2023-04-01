import torch
#from model import Model

model = torch.hub.load('lbleal1/torch-hub-test', 
                       'model', pretrained=True)
print(model)