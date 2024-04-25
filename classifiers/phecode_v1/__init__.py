import argparse
import numpy as np
import pandas as pd
import sys
import os
import torch
import pathlib
import torch.nn as nn
from . import classes
import monai


def get_cache_folder():
    return os.path.expanduser(os.path.join("~", ".cache", "latentshift/"))

thisfolder = os.path.dirname(__file__)

class PheCodeClassifier(nn.Module):

    def __init__(self):
        super(PheCodeClassifier, self).__init__()

        self.resolution = 224
        
        _MODEL_PATH = (
            pathlib.Path(get_cache_folder()) / "densenet_121_phecodes_expanded.pt"
        )
    
        self.model = monai.networks.nets.densenet.densenet121(
            spatial_dims=3, in_channels=1, out_channels=1692
        )
        self.model.load_state_dict(torch.load(_MODEL_PATH, map_location=torch.device('cpu')))
    
        self.upsample = torch.nn.Upsample(size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        
    
        self.csv = _map_to_phenotypes()
        self.targets = self.csv.phecode_str.tolist()

    def forward(self, x):
        #expecting input between 0,1, adapt to range of this model
        x = ((x*4)-2)
        x = self.upsample(x[0].swapaxes(1,3)).swapaxes(1,3)[None,...]
        assert x.shape[-3:-1] == (self.resolution,self.resolution)
        return self.model(x)
    
    

def _map_to_phenotypes():

    _ICD10_PHECODE_PATH = (
        pathlib.Path(thisfolder) / "data" / "Phecode_map_v1_2_icd10cm_beta.csv"
    )
        
    _ICD9_PHECODE_PATH = pathlib.Path(thisfolder) / "data" / "phecode_icd9_rolled.csv"

    
    class_floats = classes.DENSENET_121_PHECODES_EXPANDED
    class_strings = [str(x) for x in class_floats]

    icd10_phecodes = pd.read_csv(_ICD10_PHECODE_PATH)
    icd10_phecodes = icd10_phecodes[["phecode", "phecode_str"]]
    icd10_phecodes.loc[:, "phecode"] = icd10_phecodes.loc[:, "phecode"].astype(str)
    icd10_phecodes.loc[:, "phecode"] = icd10_phecodes.loc[:, "phecode"].apply(
        lambda x: x.rstrip("0").rstrip(".") if "." in x else x
    )
    icd10_phecodes = icd10_phecodes.drop_duplicates(subset=["phecode"])

    # do same for icd9_phecodes
    icd9_phecodes = pd.read_csv(_ICD9_PHECODE_PATH)
    icd9_phecodes = icd9_phecodes[["PheCode", "Phenotype"]]
    icd9_phecodes.columns = ["phecode", "phecode_str"]
    icd9_phecodes.loc[:, "phecode"] = icd9_phecodes.loc[:, "phecode"].astype(str)
    icd9_phecodes.loc[:, "phecode"] = icd9_phecodes.loc[:, "phecode"].apply(
        lambda x: x.rstrip("0").rstrip(".") if "." in x else x
    )
    icd9_phecodes = icd9_phecodes.drop_duplicates(subset=["phecode"])
    icd_phecodes = pd.concat([icd10_phecodes, icd9_phecodes])
    icd_phecodes = icd_phecodes.drop_duplicates(subset=["phecode"])

    class_strings_df = pd.DataFrame(class_strings, columns=["phecode"])
    class_strings_df = class_strings_df.merge(icd_phecodes, how="inner", on="phecode")
    #class_strings_df.loc[:, "prediction"] = predictions[0, :]
    return class_strings_df
    #print("Predictions:")
    #print(class_strings_df.to_string(index=False))