import torch
import torch.nn as nn
from omegaconf import OmegaConf
import taming.models.vqgan
import torchvision
import numpy as np
import latentshift
import os

baseurl = 'https://github.com/ieee8023/ct-counterfactuals/releases/download/init/'

class VQGAN(nn.Module):
    
    def __init__(self, weights=None, ckpt_path=None, yaml_path=None, download=False, resize=None, cache_dir=None):
        super().__init__()
        
        if (weights is None) and ((ckpt_path is None) and (yaml_path is None)):
            raise Exception('Must specify (weights) or (ckpt_path and yaml_path)')

        if not weights is None:
            if weights == '2023-12-25T10-26-40_ct2_vqgan256_sddd':
                resize = 256
            
            if cache_dir is None:
                cache_dir = latentshift.utils.get_cache_folder()
            ckpt_path = cache_dir + weights + '.ckpt'
            yaml_path = cache_dir + weights + '.yaml'
            if (not os.path.isfile(ckpt_path)) or (not os.path.isfile(yaml_path)):
                if download:
                    latentshift.utils.download(baseurl + weights, ckpt_path)
                    latentshift.utils.download(baseurl + weights, yaml_path)
                else:
                    print("No weights found, specify download=True to download them.")
        
        self.resize = resize
        c = OmegaConf.load(yaml_path)
        
        self.config = c['model']['params']
        self.resolution = self.config['ddconfig']['resolution']
        
        vqmodel = taming.models.vqgan.VQModel(**self.config)
        
        if ckpt_path is not None:
            a = torch.load(ckpt_path, map_location=torch.device('cpu'))
            vqmodel.load_state_dict(a['state_dict'])
        
        self.model = vqmodel
    
    def encode(self, x):

        # input range is 0,1 but this model expects [-1, 1]
        x = ((x*2)-1)
        
        self.input_size = x.shape[3]
        if not self.resize is None:
            x = torchvision.transforms.Resize(self.resize)(x)
            
        return self.model.encode(x)[0]
    
    def decode(self, z, image_shape=None):
        xp = self.model.decode(z)
        if not self.resize is None:
            xp = torchvision.transforms.Resize(self.input_size)(xp)

        # the model should output between 0,1
        xp = ((xp+2)/4)
        
        return xp
    
    def forward(self, x):
        return self.decode(self.encode(x))


class SliceAECrop:
    def __init__(self, ae, start, end, limit_to_slices=False):
        self.ae = ae
        self.start = start
        self.end = end
        self.limit_to_slices = limit_to_slices
        
    def encode(self,x):
        """Slice from the """
        self.x = x
        zz = x[:,:,:,:,self.start:self.end].transpose(-1,0)[...,0]
        return self.ae.encode(zz)

    def decode(self,zz):
        xxp = self.ae.decode(zz)
        if self.limit_to_slices:
            return  xxp[...,None].transpose(-1,0)
        else:
            return torch.cat([self.x[:,:,:,:,:self.start], xxp[...,None].transpose(-1,0), self.x[:,:,:,:,self.end:]], 4)


class SliceAEFull:
    """Encodes the entire volume but restricts the gradient to only decoding between some range"""
    def __init__(self, ae, start, end, batch_size = 16):
        self.ae = ae
        self.start = start
        self.end = end
        self.batch_size = batch_size
        
    def encode(self, x):
        """Slice from the """
        zz = x.transpose(-1,0)[...,0]
        with torch.no_grad():
            zs = [self.ae.encode(zz[i:i+self.batch_size]) for i in np.arange(0,len(zz),self.batch_size)]
        zs = torch.cat(zs)
        return zs

    def decode(self, zz):

        with torch.no_grad():
            xx = [self.ae.decode(zz[i:i+self.batch_size]) for i in np.arange(0,len(zz),self.batch_size)]
            
        xx = torch.cat(xx)
        xx[self.start:self.end] = self.ae.decode(zz[self.start:self.end])
        
        return xx[...,None].transpose(-1,0)







