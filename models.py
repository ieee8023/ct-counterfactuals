import torch
import torch.nn as nn
from omegaconf import OmegaConf
import taming.models.vqgan

class VQGAN(nn.Module):
    
    def __init__(self, ckpt_path, yaml_path, download=False):
        super().__init__()
        
        c = OmegaConf.load(yaml_path)
        
        self.config = c['model']['params']
        self.resolution = self.config['ddconfig']['resolution']
        
        vqmodel = taming.models.vqgan.VQModel(**self.config)
        
        if ckpt_path is not None:
            a = torch.load(ckpt_path, map_location=torch.device('cpu'))
            vqmodel.load_state_dict(a['state_dict'])
        
        self.model = vqmodel
        
        #self.upsample = torch.nn.Upsample(size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
    
    def encode(self, x):
        #x = (x/1024)
        #x = self.upsample(x)
        return self.model.encode(x)[0]
    
    def decode(self, z, image_shape=None):
        xp = self.model.decode(z)
        #xp = (xp*1024)
        #xp = torch.clip(xp,-1024,1024)
        return xp
    
    def forward(self, x):
        return self.decode(self.encode(x))