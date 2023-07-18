import pandas as pd
import nibabel as nib
from tqdm.autonotebook import tqdm

class TotalSegmenter_Dataset():
    def __init__(self, path, transform=None):
        super().__init__()
        
        self.path = path
        self.transform = transform

        self.raw_csv = pd.read_csv(self.path + 'meta.csv', delimiter=';')
        
        
        sizes = []
        #for image_id in tqdm(self.raw_csv.image_id):
        for i, row in tqdm(self.raw_csv.iterrows(), total=self.raw_csv.shape[0]):
            g = nib.load(path + row.image_id + "/ct.nii.gz")
            meta = {
                'frame': list(range(g.shape[2])),
                'frame_height':g.shape[0],
                'frame_width': g.shape[1],
            }
            meta.update(row)
            meta.update(dict(g.header))
            sizes.append(meta)
            
        self.csv = pd.DataFrame(sizes)
        self.csv = self.csv.explode('frame')
        

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        
        sample = self.csv.iloc[idx]
        g = nib.load(self.path + sample['image_id'] + "/ct.nii.gz")
        volume = g.get_fdata()
        return volume[:,:,sample['frame']]
