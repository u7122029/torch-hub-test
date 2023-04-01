import torch
import torch.nn as nn
from typing import Any
from torch.hub import load_state_dict_from_url

__all__ = ['Model', 'model']

model_urls = {
    'model': 'https://github.com/lbleal1/torch-hub-test/raw/main/model_resources/iris_classifier_params.pt',
}

# model
class Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, output_size=3)->None:
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Softmax(dim=1)(x)
        return x
    
def model(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Model:
    r"""
    
    """
    model = Model(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['model'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model