import torch
import torch.nn as nn
from typing import Any

__all__ = ['Model', 'model']

# model
class Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, output_size=3):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Softmax(dim=1)(x)
        return x
    
def model(pretrained: bool = False, **kwargs: Any) -> Model:
    r"""
    
    """
    model = Model(**kwargs)
    if pretrained:
        path = '..\model_resources\iris_classifier.pt'
        model = torch.load(path)
    return model