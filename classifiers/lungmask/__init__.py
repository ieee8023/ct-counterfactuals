import argparse
import numpy as np
import pandas as pd
import sys
import os
import torch
import pathlib
import torch.nn as nn
from .resunet import UNet


def get_cache_folder():
    return os.path.expanduser(os.path.join("~", ".cache", "latentshift/"))

thisfolder = os.path.dirname(__file__)



class LungMaskSegmenter(nn.Module):

    def __init__(self):
        super(LungMaskSegmenter, self).__init__()

        self.resolution = 224
        
        _MODEL_PATH = (
            pathlib.Path(get_cache_folder()) / "lungmask-unet_r231-d5d2fc3d.pth"
        )
        state_dict = torch.load(_MODEL_PATH, map_location=torch.device('cpu'))

        self.n_classes = len(list(state_dict.values())[-1])
     
        self.model = UNet(
            n_classes=self.n_classes,
            padding=True,
            depth=5,
            up_mode="upsample",
            batch_norm=True,
            residual=False,
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.eval()
    
        self.targets = ['Lung']

    def forward(self, x):
        if len(x.shape) == 5:
            output = torch.stack([self.model(x[...,i].cuda()) for i in range(x.shape[-1])], 4)
            return torch.sigmoid(output)
        else:
            return torch.sigmoid(self.model(x))
    

