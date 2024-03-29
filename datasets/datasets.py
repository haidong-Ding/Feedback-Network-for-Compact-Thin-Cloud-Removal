import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage import io, transform
import os
import numpy as np
import random
import torchvision


class RICEDataset(Dataset):
    def __init__(self, root_dir,  
                       transform=None,
                       train=True
                       ):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
             train: train or test dataset or test
        """
        if train:
            data_list = root_dir + 'train.txt'
        else:
            data_list = root_dir + 'test.txt'
        with open(data_list) as f:
            contents = f.readlines()
            image_files = [i.strip() for i in contents]
        
        self.image_files = image_files
       
        self.root_dir = root_dir
        self.transform = transform        
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_name = self.image_files[idx]
        
        cloud_image = Image.open(self.root_dir + 'cloud/' + image_name).convert('RGB')
        ref_image = Image.open(self.root_dir + 'label/' + image_name).convert('RGB')
        
        if self.transform:
            cloud_image = self.transform(cloud_image)
            ref_image = self.transform(ref_image)
                       
        return {'cloud_image': cloud_image, 'ref_image': ref_image, 'image_name':image_name}
        
        
class WHUDataset(Dataset):
    def __init__(self, root_dir,  
                       transform=None,
                       train=True
                       ):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
             train: train or test dataset or test
        """
        if train:
            data_list = root_dir + 'train.txt'
        else:
            data_list = root_dir + 'test.txt'
        with open(data_list) as f:
            contents = f.readlines()
            image_files = [i.strip() for i in contents]
        
        self.image_files = image_files
       
        self.root_dir = root_dir
        self.transform = transform        
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_name = self.image_files[idx]
        
        cloud_image = Image.open(self.root_dir + 'cloud/' + image_name).convert('RGB')
        ref_image = Image.open(self.root_dir + 'clear/' + image_name).convert('RGB')
        
        if self.transform:
            cloud_image = self.transform(cloud_image)
            ref_image = self.transform(ref_image)
                       
        return {'cloud_image': cloud_image, 'ref_image': ref_image, 'image_name':image_name}
        

class CloudRemovalDataset(Dataset):
    def __init__(self, root_dir, 
                       crop=False, 
                       crop_size=256, 
                       rotation=False, 
                       color_augment=False, 
                       transform=None, 
                       train = True):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        if train:
            data_list = root_dir + 'train.txt'
        else:
            data_list = root_dir + 'test.txt'

        with open(data_list) as f:
            contents = f.readlines()
            image_files = [i.strip() for i in contents]
        
        self.image_files = image_files
       
        self.root_dir = root_dir
        self.transform = transform        
        self.crop = crop
        self.crop_size = crop_size
        self.rotation = rotation
        self.color_augment = color_augment
        self.rotate90 = transforms.RandomRotation(90)  
        self.rotate45 = transforms.RandomRotation(45)    
        self.train = train

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_name = self.image_files[idx]
          
        cloud_image = Image.open(self.root_dir + 'cloud/' + image_name).convert('RGB')
        ref_image = Image.open(self.root_dir + 'reference/' + image_name).convert('RGB')

        if self.rotation:
            degree = random.choice([90, 180, 270])
            cloud_image = transforms.functional.rotate(cloud_image, degree) 
            ref_image = transforms.functional.rotate(ref_image, degree)

        if self.color_augment:
            cloud_image = transforms.functional.adjust_gamma(cloud_image, 1)
            ref_image = transforms.functional.adjust_gamma(ref_image, 1)                           
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            cloud_image = transforms.functional.adjust_saturation(cloud_image, sat_factor)
            ref_image = transforms.functional.adjust_saturation(ref_image, sat_factor)
            
        if self.transform:
            cloud_image = self.transform(cloud_image)
            ref_image = self.transform(ref_image)

        if self.crop:
            W = cloud_image.size()[1]
            H = cloud_image.size()[2] 

            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]
            
            cloud_image = cloud_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            ref_image = ref_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
                       

        if self.train:
            return {'cloud_image': cloud_image, 'ref_image': ref_image}
        else:
            return {'cloud_image': cloud_image, 'ref_image': ref_image, 'image_name': image_name}


class MultiSpectralDataset(Dataset):
    def __init__(self, root_dir, crop=True, crop_size=256, rotation=False, color_augment=False, transform=None, train=True):
        super(MultiSpectralDataset, self).__init__()
        """
        Args:
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        if train:
            data_list = root_dir + 'train.txt'
        else:
            data_list = root_dir + 'test.txt'

        with open(data_list, 'r') as f:
            contents = f.readlines()
            image_files = [i.strip() for i in contents]

        self.image_files = image_files

        self.root_dir = root_dir
        self.crop = crop
        self.crop_size = crop_size
        self.rotation = rotation
        self.color_augment = color_augment
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        # cloudy image
        image_name = self.image_files[idx]
        image = Image.open(self.root_dir + 'cloud/951/' + image_name).convert('RGB')
        
        # clear image
        label_name = image_name
        if image_name[-5] == 'c':
            label_name = label_name[:-5] + 'b' + label_name[-4:]
        label = Image.open(self.root_dir + 'clear/951/' + label_name).convert('RGB')

        if self.color_augment:
            image = transforms.functional.adjust_gamma(image, 1)
            label = transforms.functional.adjust_gamma(label, 1)
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            image = transforms.functional.adjust_saturation(image, sat_factor)
            label = transforms.functional.adjust_saturation(label, sat_factor)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            
        if self.crop:
            W = image.size()[1]
            H = image.size()[2] 

            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]
            
            image = image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            label = label[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]

        # Random Rotation
        if self.rotation:
            aug = random.randint(0, 8)
            if aug == 1:
                image = image.flip(1)
                label = label.flip(1)
            elif aug == 2:
                image = image.flip(2)
                label = label.flip(2)
            elif aug == 3:
                image = torch.rot90(image, dims=(1, 2))
                label = torch.rot90(label, dims=(1, 2))
            elif aug == 4:
                image = torch.rot90(image, dims=(1, 2), k=2)
                label = torch.rot90(label, dims=(1, 2), k=2)
            elif aug == 5:
                image = torch.rot90(image, dims=(1, 2), k=3)
                label = torch.rot90(label, dims=(1, 2), k=3)
            elif aug == 6:
                image = torch.rot90(image.flip(1), dims=(1, 2))
                label = torch.rot90(label.flip(1), dims=(1, 2))
            elif aug == 7:
                image = torch.rot90(image.flip(2), dims=(1, 2))
                label = torch.rot90(label.flip(2), dims=(1, 2))

        if self.train:
            return {'cloud_image': image, 'ref_image': label}
        else:
            return {'cloud_image': image, 'ref_image': label, 'image_name': image_name}
            
        
