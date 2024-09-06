import pandas as pd
import torchvision
import torchxrayvision as xrv
from tqdm.autonotebook import tqdm
import numpy as np
import os
import skimage
import glob

import monai
import torch
import os
from tqdm import tqdm
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord, 
    CenterSpatialCropd,
    Rotate90,
    Flip,
)
from torch.utils.data import Dataset


class KeyTransform:
    def __init__(self, keys, transform):
        self.keys = keys
        self.transform = transform
        
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.transform(data[key])
                return data
            else:
                raise Exception(f'Missing key {self.key}')


class NIFTI_Dataset(Dataset):
    def __init__(self, path, transforms_image=None, resolution=256):

        self.path = path
        if transforms_image is None:
            # Initialize the image preprocessing transforms
            transforms_image = Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
                    ScaleIntensityRanged(
                        keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
                    ),
                    SpatialPadd(keys=["image"], spatial_size=[resolution, resolution, -1]),
                    CenterSpatialCropd(
                        roi_size=[resolution, resolution, -1],
                        keys=["image"],
                    ),
                    KeyTransform(keys=["image"], transform=Rotate90(k=1, spatial_axes=(0,1))),
                    KeyTransform(keys=["image"], transform=Flip(spatial_axis=1)),
                    ToTensord(keys=["image"]),
                ]
            )
        
        image_paths = glob.glob(self.path + "*.nii*")
        data_list = [{"image": image_path, "image_path": image_path} for image_path in image_paths]
        self.dataset = monai.data.Dataset(data_list, transform=transforms_image)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



def window_level(x, lower, upper):
    # abdomen, 50, 400
    lower = lower/2048 + 0.5
    upper = upper/2048 + 0.5
    print(lower, upper)
    x = np.minimum(x, upper)
    x = np.maximum(x, lower)
    return x





def _get_data(datasets_str, size=512, transform=None, path="/home/users/joecohen/scratch/data-cts/", merge=True):
    """This function is deprecated for now"""
    datasets = []
    
    if "total" in datasets_str:
        datasets.append(TotalSegmenter_Dataset(
            path + 'Totalsegmentator_dataset/', 
            transform=torchvision.transforms.Compose(
                [ResizeScale(size/256)] + 
                ([transform] if not transform is None else [])
            )
        ))
        # because total segmenter dataset is half size and the other ones are not

    if "deeplesion" in datasets_str:
        datasets.append(DeepLesion_Dataset(
            path + 'DeepLesion/', 
            transform=torchvision.transforms.Compose(
                [ResizeScale(size/512)] + 
                ([transform] if not transform is None else [])
            )
        ))

    if "luna16" in datasets_str:
        datasets.append(LUNA16_Dataset(
            path + 'LUNA16/', 
            transform=torchvision.transforms.Compose(
                [ResizeScale(size/512)] + 
                ([transform] if not transform is None else [])
            )
        ))
        
    if merge:
        dmerge = xrv.datasets.Merge_Dataset(datasets)
        return dmerge
    else:
        return datasets

class RandomCrop:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def __call__(self, img):
        assert img.shape[0] >= self.height
        assert img.shape[1] >= self.width
        x = np.random.randint(0, img.shape[1] - self.width)
        y = np.random.randint(0, img.shape[0] - self.height)
        img = img[y:y+self.height, x:x+self.width]
        return img


class Resize:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        return skimage.transform.resize(
            img,
            (self.size, self.size), 
            anti_aliasing=True,
        )

class ResizeScale:
    def __init__(self, scale):
        self.scale = scale
        
    def __call__(self, img):
        return skimage.transform.resize(
            img,
            (img.shape[0]*self.scale, img.shape[1]*self.scale), 
            anti_aliasing=True,
        )


class _TotalSegmenter_Dataset():
    """This dataset is deprecated for now"""
    def __init__(self, path, transform=None):
        super().__init__()
        
        self.path = path
        self.cache_dir = self.path + '/cache/'
        self.transform = transform
        self.raw_csv = pd.read_csv(self.path + 'meta.csv', delimiter=';')
        
        self.broken_zips = ['s0864']
        self.raw_csv = self.raw_csv[~self.raw_csv.image_id.isin(self.broken_zips)]
        
        metadata_path = self.path + 'meta_frame.csv.gz'
        if not os.path.exists(metadata_path):
            print('creating frame metadata at ' + metadata_path)
            self.create_meta(metadata_path)
        self.csv = pd.read_csv(metadata_path) 
        
        self.create_cache()
        
        self.pathologies = []
        self.labels = np.zeros([len(self.csv),0])

            
    def create_meta(self, filename):
        import nibabel as nib
        sizes = []
        for i, row in tqdm(self.raw_csv.iterrows(), total=self.raw_csv.shape[0]):
            g = nib.load(self.path + row.image_id + "/ct.nii.gz")
            meta = {
                'frame': list(range(g.shape[2])),
                'frame_height':g.shape[0],
                'frame_width': g.shape[1],
            }
            meta.update(row)
            meta.update(dict(g.header))
            sizes.append(meta)
            
        csv = pd.DataFrame(sizes).explode('frame')
        csv.to_csv(filename, index=None)
        

    def create_cache(self):
        import nibabel as nib
        os.makedirs(self.cache_dir, exist_ok=True)
        
        for image_id, row in tqdm(self.csv.groupby('image_id')):
            if not os.path.exists(self.cache_dir + f'/{image_id}_0.npz'):
                
                path = self.path + image_id + "/ct.nii.gz"
                g = nib.load(path)
                volume = g.get_fdata()

                for frame_idx in row.frame:
                    cache_path = self.cache_dir + f'/{image_id}_{frame_idx}.npz'
                    np.savez_compressed(cache_path, volume[:,:,frame_idx])
    
    
    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __repr__(self):
        return self.string()
        
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        
        item = {}
        sample = self.csv.iloc[idx]
        
        cache_path = self.cache_dir + f'/{sample["image_id"]}_{sample["frame"]}.npz'

        try:
            item['img'] = np.load(cache_path)['arr_0']
            
            item['img'] = item['img']/1024.0
    
            if self.transform:
                item['img'] = self.transform(item['img'])
            return item
        except Exception as e:
            print(f'Issue with {cache_path}')
            raise e


class _DeepLesion_Dataset():
    """This dataset is deprecated for now"""
    def __init__(self, path, transform=None, img_folder_path='/Images_png/Images_png/'):
        super().__init__()
        
        self.path = path
        self.img_folder_path = img_folder_path
        self.transform = transform
        self.raw_csv = pd.read_csv(self.path + 'DL_info.csv')
        self.csv = self.raw_csv.copy()

        def expand(x):
            s = x.split(', ')
            return np.arange(int(s[0]),int(s[1])).tolist()
        
        self.csv['slice'] = self.csv.Slice_range.apply(expand)
        self.csv = self.csv.explode('slice')

        self.csv = self.csv.groupby(['Patient_index','Study_index','Series_ID','slice']).first().reset_index()

        self.csv['frame_height'] = self.csv.Image_size.apply(lambda x: int(x.split(', ')[0]))
        self.csv['frame_width'] = self.csv.Image_size.apply(lambda x: int(x.split(', ')[1]))
        
        self.pathologies = []
        self.labels = np.zeros([len(self.csv),0])
    
    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))
        
    def __repr__(self):
        return self.string()

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        
        item = {}
        row = self.csv.iloc[idx]

        path = self.path + self.img_folder_path + f'{row.Patient_index:06d}_{row.Study_index:02d}_{row.Series_ID:02d}/{row.Key_slice_index:03d}.png'
        
        try:
            item['img'] = skimage.io.imread(path)
    
            item['img'] = skimage.transform.rotate(item['img'],-90, preserve_range=True)
    
            # normalize 
            item['img'] = ((item['img']-31536.0)/(34686.0-31536.0))
            item['img'] = (item['img']*2.0)-1.0
    
            if self.transform:
                item['img'] = self.transform(item['img'])
            return item
        except Exception as e:
            print(f'Issue with {path}')
            raise e


class _LUNA16_Dataset():
    """This dataset is deprecated for now"""
    def __init__(self, path, transform=None):
        super().__init__()
        
        self.path = path
        self.cache_dir = self.path + '/cache/'
        self.transform = transform
        self.raw_csv = pd.DataFrame(glob.glob(f'{path}/*.mhd'),columns=['filename'])

        if len(self.raw_csv) == 0:
            raise Exception('No mhd files found. The files may not have been unzipped.')
        
        self.raw_csv['filename'] = self.raw_csv.filename.apply(os.path.basename)
        
        self.metadata_path = self.path + 'meta_frame.csv.gz'
        if not os.path.exists(self.metadata_path):
            print('creating frame metadata at ' + self.metadata_path)
            self.create_meta(self.metadata_path)
        self.csv = pd.read_csv(self.metadata_path) 
        
        self.create_cache()
        
        self.pathologies = []
        self.labels = np.zeros([len(self.csv),0])

            
    def create_meta(self, filename):
        sizes = []
        for i, row in tqdm(self.raw_csv.iterrows(), total=self.raw_csv.shape[0]):
            path = self.path + row.filename
            g = skimage.io.imread(path, plugin='simpleitk')
            meta = {
                'frame': list(range(g.shape[0])),
                'frame_height':g.shape[1],
                'frame_width': g.shape[2],
            }
            meta.update(row)
            sizes.append(meta)
            
        csv = pd.DataFrame(sizes).explode('frame')
        csv.to_csv(filename, index=None)
        

    def create_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        
        for filename, row in tqdm(self.csv.groupby('filename')):
            all_exist = True
            for frame_idx in row.frame:
                path_to_check = f'{self.cache_dir}/{filename}_{frame_idx}.npz'
                if not all_exist:
                    break # we found an error already
                elif not os.path.exists(path_to_check):
                    all_exist = False
                    if frame_idx != 0: # if not the first one
                        print(f'Missing cache file {path_to_check}. Will recompute.')
                    
            if not all_exist:
                path = self.path + filename
                g = skimage.io.imread(path, plugin='simpleitk')

                for frame_idx in row.frame:
                    cache_path = self.cache_dir + f'/{filename}_{frame_idx}.npz'
                    img = g[frame_idx,:,:]
                    img = skimage.transform.rotate(img,-90, preserve_range=True)
                    np.savez_compressed(cache_path, img)
    
    
    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __repr__(self):
        return self.string()
        
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        
        item = {}
        sample = self.csv.iloc[idx]
        
        cache_path = self.cache_dir + f'/{sample["filename"]}_{sample["frame"]}.npz'
        try:
            item['img'] = np.load(cache_path)['arr_0']
            
            item['img'] = item['img']/1024.0
    
            if self.transform:
                item['img'] = self.transform(item['img'])
            return item
        except Exception as e:
            print(f'Issue with {cache_path}')
            raise e

