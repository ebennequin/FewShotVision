# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import json
import os
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms


identity = lambda x:x
# TODO: why not extend torch.utils.data.Dataset ?
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity, shallow=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        if shallow: # We return a reduced dataset
            self.meta['image_names'] = self.meta['image_names'][:256] # TODO: shallow not applied to SetDataset
            self.meta['image_labels'] = self.meta['image_labels'][:256]
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity): #TODO: why this assignment to transform
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(torch.utils.data.Sampler):
    '''
    Samples elements randomly in episodes of defined shape
    '''
    def __init__(self, n_classes, n_way, n_episodes):
        '''

        Args:
            n_classes (int): number of classes in the dataset
            n_way (int): number of classes in an episode
            n_episodes (int): number of episodes
        '''
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
