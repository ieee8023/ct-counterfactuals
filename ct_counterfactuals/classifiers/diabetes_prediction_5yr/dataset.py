import pandas as pd
import numpy as np
import nibabel as nib
import torch
import os
import sys
from monai.transforms import CenterSpatialCrop, SpatialPad
from torch.utils.data import Dataset

class DiabetesPredictionDataset(Dataset):
    def __init__(self, csv_file, split_type):
        """
        Args:
            csv_file (string): Path to the csv file with all the labels and filenames.
            split_type (string): One of ['train', 'val', 'test'].
        """
        self.data_frame = pd.read_csv(csv_file)
        # Filter the data_frame based on diabetes label (dm) and split type
        if split_type == 'train':
            splits = [0, 1, 2]
        elif split_type == 'val':
            splits = [3]
        elif split_type == 'test':
            splits = [4]
        self.data_frame = self.data_frame[(self.data_frame.dm.isin([0,1])) & (self.data_frame.split.isin(splits))]
        accession_numbers = self.data_frame['ct_filename'].apply(lambda x: x.split('-')[1].split('_')[0])

        # add to data_frame
        self.data_frame['accession'] = accession_numbers

        # now, load all file names in '/dataNAS/people/lblankem/ct_rrg_v1/scans/2/all/{accession}_0000.nii.gz' and extract those accessions
        # that are in the data_frame

        file_names = os.listdir('/dataNAS/people/lblankem/ct_rrg_v1/scans/2/all/')
        print(len(file_names))
        accessions = [file_name.split('_')[0] for file_name in file_names]
        # make a dataframe with accessions
        accessions = pd.DataFrame(accessions, columns=['accession'])
        # merge so that we only have accessions that are in the data_frame
        self.data_frame = pd.merge(self.data_frame, accessions, on='accession', how='inner')

        print(f"Number of scans in {split_type} split: {len(self.data_frame)}")
        # print the number of patients with diabetes and without
        print(f"Number of scans with diabetes: {len(self.data_frame[self.data_frame['dm'] == 1])}")
        print(f"Number of scans without diabetes: {len(self.data_frame[self.data_frame['dm'] == 0])}")
        # flush 
        sys.stdout.flush()

    def __len__(self):
        return len(self.data_frame)


    def __getitem__(self, idx):
        # Load the image as a numpy array
        ct_filename = self.data_frame.iloc[idx]['ct_filename']
        accession = ct_filename.split('-')[1].split('_')[0]
        nifti_img = nib.load(f'/dataNAS/people/lblankem/ct_rrg_v1/scans/2/all/{accession}_0000.nii.gz')
        img_data = np.array(nifti_img.dataobj, dtype=np.float32)
        
        # Scale the image values
        img_data = img_data / 1024.0

        # take every other slice in the z direction, in the case where the image spacing is originally 1.5mm in the z direction
        img_data = img_data[:, :, ::2]

        # Define the target size
        target_size = [250, 250, 150]

        # Check if padding is needed
        if np.any(np.array(img_data.shape) < target_size):
            pad_width = [(0, max_size - img_size) if img_size < max_size else (0, 0) for img_size, max_size in zip(img_data.shape, target_size)]
            img_data = np.pad(img_data, pad_width, mode='constant')

        # Check if cropping is needed
        if np.any(np.array(img_data.shape) > target_size):
            crop_width = [(img_size - max_size) // 2 if img_size > max_size else 0 for img_size, max_size in zip(img_data.shape, target_size)]
            img_data = img_data[crop_width[0]:crop_width[0]+target_size[0], crop_width[1]:crop_width[1]+target_size[1], crop_width[2]:crop_width[2]+target_size[2]]

        # save to nifti
        nifti_img = nib.Nifti1Image(img_data, affine=nifti_img.affine)

        # Add the channel dimension
        img_data = img_data[np.newaxis, :, :, :]

        # Convert to tensor
        img_data = torch.from_numpy(img_data)

        # Get the diabetes label
        dm_label = self.data_frame.iloc[idx]['dm']
        dm_label = torch.tensor(dm_label)

        return img_data, dm_label







