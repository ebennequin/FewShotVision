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


class DetectionTaskSampler(torch.utils.data.Sampler):
    '''
    Samples elements in detection episodes of defined shape.
    '''
    def __init__(self, data_source, n_way, n_support, n_query, n_episodes):
        '''

        Args:
            data_source (torch.utils.data.Dataset): source dataset
            n_way (int): number of different classes in a detection class
            n_support (int): number of images in the support set with an instance of one class,
            for each of the n_way classes
            n_query (int): number of images in the query set with an instance of one class,
            for each of the n_way classes
            n_episode (int): number of episodes per epoch
        '''
        self.data_source = data_source
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes

        self.images_per_label = self._get_images_per_label()
        self.label_list = list(self.images_per_label.keys())

    def _get_images_per_label(self):
        '''

        Returns:
            dict: each key maps to a list of the images which contain at least one target which label is the key
        '''
        images_per_label={}

        for index in range(len(self.data_source)):
            targets = self.data_source[index][2]
            for target in targets:
                label = int(target[1])
                if label not in images_per_label.keys():
                    images_per_label[label] = []
                if len(images_per_label[label]) == 0 or images_per_label[label][-1] != index:
                    images_per_label[label].append(index)
            if index % 100 == 0:
                print('{}/{} images considered'.format(index, len(self.data_source)))
        return images_per_label

    def _sample_labels(self):
        '''

        Returns:
            ndarray: n_way labels sampled at random from all available labels
        '''
        labels = np.random.choice(self.label_list, self.n_way, replace=False)
        return labels

    def _sample_images_from_labels(self, labels):
        '''

        Args:
            labels (ndarray): labels from which images will be sampled

        Returns:
            torch.Tensor: indices of images constituting an episode
        '''
        #TODO: images can appear twice, and some labels don't have enough images
        images_indices = []
        for label in labels:
            images_from_label = np.random.choice(
                self.images_per_label[label],
                self.n_support+self.n_query,
                replace=False
            )
            images_indices.extend(images_from_label)
        return torch.IntTensor(images_indices)

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            labels = self._sample_labels()
            yield self._sample_images_from_labels(labels)
