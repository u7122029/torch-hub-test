dependencies = ['torch']

import torch

def custom_model():
    """ # This docstring shows up in hub.help()
    Loads a model from a path
    and returns the model 
    """
    # load
    path = 'lbleal1/load_custom_model/iris_classifier.pt'
    model = torch.load(path)
    return model


