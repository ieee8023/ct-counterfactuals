import pandas as pd
import nibabel as nib
import torchxrayvision as xrv
from tqdm.autonotebook import tqdm
import numpy as np
import os


def get_data(datasets_str, transform=None, path="/home/users/joecohen/scratch/data-scratch/Totalsegmentator_dataset/", merge=True):
    
    datasets = []
    
    if "total" in datasets_str:
        datasets.append(TotalSegmenter_Dataset(path, transform))
    
    if merge:
        dmerge = xrv.datasets.Merge_Dataset(datasets)
        return dmerge
    else:
        return datasets

    
class TotalSegmenter_Dataset():
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
        os.makedirs(self.cache_dir, exists_ok=True)
        
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

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        
        item = {}
        sample = self.csv.iloc[idx]
        
        cache_path = self.cache_dir + f'/{sample["image_id"]}_{sample["frame"]}.npz'
        item['img'] = np.load(cache_path)['arr_0']

        if self.transform:
            item['img'] = self.transform(item['img'])
        return item
