# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import json
import os
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms


# TODO: why not extend torch.utils.data.Dataset ?
class SimpleDataset:
    '''
    Defines a regular dataset of images
    '''
    def __init__(self, data_file, transform, shallow=False):
        '''

        Args:
            data_file (str): path to JSON file defining the data
            transform (torchvision.transforms.Compose): transformations to be applied to the images
            shallow (bool): whether to create only a small dataset (for quick code testing)
        '''
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        if shallow:  # We return a reduced dataset
            self.meta['image_names'] = self.meta['image_names'][:256]  # TODO: shallow not applied to SetDataset
            self.meta['image_labels'] = self.meta['image_labels'][:256]
        self.transform = transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        return img, self.meta['image_labels'][i]

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    '''
    Defines a dataset splitted in subsets.
    Item of index i is a torch.utils.DataLoader object constructed from the SubDataset corresponding to label i.
    '''
    def __init__(self, data_file, batch_size, transform):
        '''

        Args:
            data_file (str): path to JSON file defining the data
            batch_size (int): number of images per class in an episode
            transform (torchvision.transforms.Compose): transformations to be applied to the images
        '''
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.label_list = np.unique(self.meta['image_labels']).tolist()

        self.images_per_label = {}
        for label in self.label_list:
            self.images_per_label[label] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.images_per_label[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for label in self.label_list:
            sub_dataset = SubDataset(self.images_per_label[label], label, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.label_list)


class SubDataset:
    def __init__(self, images_list, label, transform=transforms.ToTensor()):
        '''
        Defines the dataset composed by the images of a label
        Args:
            images_list (list): contains the paths to the images
            label (int): original label of the images
            transform (torchvision.transforms.Compose): transformations to be applied to the images
        '''
        self.images_list = images_list
        self.label = label
        self.transform = transform

    def __getitem__(self, i):
        image_path = os.path.join(self.images_list[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        return img, self.label

    def __len__(self):
        return len(self.images_list)


class EpisodicBatchSampler(torch.utils.data.Sampler):
    '''
    Samples elements randomly in episodes of defined shape.
    Each yielded sample is a torch.Tensor of shape (n_way)
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
