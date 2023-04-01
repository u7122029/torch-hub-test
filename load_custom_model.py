import torch

model = torch.hub.load('lbleal1/torch-hub-test', 
                       'model', pretrained=True)
print(model)