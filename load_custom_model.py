import torch

model = torch.hub.load('lbleal1/torch-hub-test', 
                       'custom_model',
                       trust_repo = True)
print(model)