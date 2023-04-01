dependencies = ['torch']

import torch
from model import Model

def custom_model():
    """ # This docstring shows up in hub.help()
    Loads a model from a path
    and returns the model 
    """
    # load
    path = 'iris_classifier.pt'
    model = torch.load(path)
    return model

def custom_model_params():
    """ # This docstring shows up in hub.help()
    Loads a model given model parameters
    and returns the model 
    """

    path = 'iris_classifier_params.pt'
    model = Model()
    model.load_state_dict(torch.load(path))
    return model

