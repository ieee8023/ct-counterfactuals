import argparse
import numpy as np
import pandas as pd
import sys
import os
import torch
import pathlib
import torch.nn as nn
import latentshift
from .resunet import UNet


class LungMaskSegmenter(nn.Module):
    """U-net(R231): This model was trained on a large and diverse dataset that covers a wide range of visual variabiliy. The model performs segmentation on individual slices, extracts right-left lung seperately includes airpockets, tumors and effusions. The trachea will not be included in the lung segmentation. https://doi.org/10.1186/s41747-020-00173-2

    ```python
    import ct_counterfactuals as ct_cf
    model = ct_cf.classifiers.lungmask.LungMaskSegmenter()
    x = torch.ones([1, 1, 224, 224, 174])
    out = model(x)
    out.shape # [1, 3, 224, 224, 1]
    
    # Channels
    #0 = No lung
    #1 = Right lung
    #2 = Left lung
    ```
    
    https://github.com/JoHof/lungmask

    Hofmanninger, J., Prayer, F., Pan, J. et al. Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. Eur Radiol Exp 4, 50 (2020). https://doi.org/10.1186/s41747-020-00173-2
    """

    def __init__(self, download=False):
        super(LungMaskSegmenter, self).__init__()

        self.resolution = 224
        
        weights_path = (
            pathlib.Path(latentshift.utils.get_cache_folder()) / "lungmask-unet_r231-d5d2fc3d.pth"
        )

        if not os.path.isfile(weights_path):
            if download:
                latentshift.utils.download(baseurl + weights, ckpt_path)
            else:
                print("No weights found, specify download=True to download them.")

        
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

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
    
        self.targets = ['No Lung', 'Right Lung', 'Left Lung']

    def forward(self, x):
        if len(x.shape) == 5: # if a volume
            output = torch.stack([self.model(x[...,i]) for i in range(x.shape[-1])], 4)
            return torch.sigmoid(output)
        else: #if a single slice
            return torch.sigmoid(self.model(x))
    

