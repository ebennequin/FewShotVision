# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision.transforms as transforms
from src.loaders import additional_transforms as add_transforms
from src.loaders.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler, DetectionTaskSampler
from src.yolov3.utils.datasets import ListDataset
from abc import abstractmethod


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):  # Returns transformation method from its String name
        if transform_type == 'ImageJitter':  # Change Brightness, Constrast, Color and Sharpness randomly
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):  # Returns composed transformation
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, path_to_data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, path_to_data_file, aug, shallow=False):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(path_to_data_file, transform, shallow=shallow)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=12, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_episode=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, path_to_data_file, aug):  # parameters that would change on train/val set
        '''

        Args:
            path_to_data_file (str): path to JSON file describing the data
            aug (bool): whether or not to perform data augmentation on the dataset

        Returns:
            DataLoader: data loader containing episodes sampled from the dataset.
            Each episode is a tuple composed of a torch.Tensor with shape (n_way, n_support+n_query, (image_dim))
            and a torch.Tensor with shape (n_way, n_support+n_query) containing the associated labels.
        '''
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(path_to_data_file, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


class DetectionSetDataManager(DataManager):
    '''
    Data Manager used for YOLOMAML
    '''
    def __init__(self, n_way, n_support, n_query, n_episode, image_size):
        '''

        Args:
            n_way (int): number of different classes in a detection class
            n_support (int): number of images in the support set with an instance of one class,
            for each of the n_way classes
            n_query (int): number of images in the query set with an instance of one class,
            for each of the n_way classes
            n_episode (int): number of episodes per epoch
            image_size (int): size of images
        '''
        super(DetectionSetDataManager).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episode = n_episode
        self.image_size = image_size

    def get_data_loader(self, path_to_data_file, path_to_images_per_label=None):
        '''

        Args:
            path_to_data_file (str): path to file containing paths to images
            path_to_images_per_label (str): path to pkl file containing images_per_label dictionary (optional)

        Returns:
            DataLoader: samples data in the shape of a detection task
        '''
        dataset = ListDataset(path_to_data_file, img_size=self.image_size)
        sampler = DetectionTaskSampler(
            dataset,
            self.n_way,
            self.n_support,
            self.n_query,
            self.n_episode,
            path_to_images_per_label,
        )
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_sampler=sampler,
                                                  num_workers=12,
                                                  collate_fn=dataset.collate_fn,
                                                  )
        return data_loader
