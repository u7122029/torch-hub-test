import torch
#from model import Model

model = torch.hub.load('lbleal1/torch-hub-test', 
                       'custom_model')
print(model)