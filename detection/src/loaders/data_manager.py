import pickle

import numpy as np
import torch

from detection.src.yolov3.utils.datasets import ListDataset


class DetectionSetDataManager():
    """
    Data Manager used for YOLOMAML
    """
    def __init__(self, n_way, n_support, n_query, n_episode, image_size):
        """

        Args:
            n_way (int): number of different classes in a detection class
            n_support (int): number of images in the support set with an instance of one class,
            for each of the n_way classes
            n_query (int): number of images in the query set with an instance of one class,
            for each of the n_way classes
            n_episode (int): number of episodes per epoch
            image_size (int): size of images (square)
        """
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episode = n_episode
        self.image_size = image_size

    def get_data_loader(self, path_to_data_file, path_to_images_per_label=None):
        """

        Args:
            path_to_data_file (str): path to file containing paths to images
            path_to_images_per_label (str): path to pkl file containing images_per_label dictionary (optional)

        Returns:
            DataLoader: samples data in the shape of a detection task
        """
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
                                                  collate_fn=dataset.collate_fn_episodic,
                                                  )
        return data_loader


def create_dict_images_per_label(data_source):
    """
            Compute and returns dictionary of images per label
            Args:
                data_source (ListDataset) : The data set containing the images
            Returns:
                dict: each key maps to a list of the images which contain at least one target which label is the key
            """
    images_per_label={}

    for index in range(len(data_source)):
        try:
            targets = data_source[index][2]
            if targets is not None:
                for target in targets:
                    label = int(target[1])
                    if label not in images_per_label:
                        images_per_label[label] = []
                    if len(images_per_label[label]) == 0 or images_per_label[label][-1] != index:
                        images_per_label[label].append(index)
                if index % 100 == 0:
                    print('{index}/{length_data_source} images considered'.format(
                        index=index,
                        length_data_source=len(data_source))
                    )
        except OSError:
            print('Corrupted image : {image_index}'.format(image_index=index))
    return images_per_label


class DetectionTaskSampler(torch.utils.data.Sampler):
    """
    Samples elements in detection episodes of defined shape.
    """
    def __init__(self, data_source, n_way, n_support, n_query, n_episodes, path_to_images_per_label=None):
        """

        Args:
            data_source (ListDataset): source dataset
            n_way (int): number of different classes in a detection class
            n_support (int): number of images in the support set with an instance of one class,
            for each of the n_way classes
            n_query (int): number of images in the query set with an instance of one class,
            for each of the n_way classes
            n_episodes (int): number of episodes per epoch
            path_to_images_per_label (str): path to a pickle file containing a dictionary of images per label
        """
        self.data_source = data_source
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes

        self.images_per_label = self.get_images_per_label(path_to_images_per_label)
        self.label_list = self.get_label_list()

    def get_images_per_label(self, path):
        """
        Returns dictionary of images per label from a file if specified or compute it from scratch
        Args:
            path (str) : path to a pickle file containing a dictionary of images per label
        Returns:
            dict: each key maps to a list of the images which contain at least one target which label is the key
        """
        if path:
            with open(path, 'rb') as dictionary_file:
                images_per_label = pickle.load(dictionary_file)
        else:
            images_per_label = create_dict_images_per_label(self.data_source)

        return images_per_label

    def get_label_list(self):
        """

        Returns:
            list: list of appropriate labels, i.e. labels that are present in at least n_support+n_query images
        """
        label_list = []
        for label in self.images_per_label:
            if len(self.images_per_label[label]) >= self.n_support + self.n_query:
                label_list.append(label)
        return label_list

    def sample_labels(self):
        """

        Returns:
            numpy.ndarray: n_way labels sampled at random from all available labels
        """
        labels = np.random.choice(self.label_list, self.n_way, replace=False)
        return labels

    def sample_images_from_labels(self, labels):
        """
        For each label in labels, samples n_support+n_query images containing at least one box associated with label
        The first n_way elements of the returned tensor will be used to determine the sampled labels
        Args:
            labels (numpy.ndarray): labels from which images will be sampled

        Returns:
            torch.Tensor: length = n_way*(1+n_support+n_query) information about the labels,
            and indices of images constituting an episode
        """
        #TODO: images can appear twice
        images_indices = list(-labels-1)
        for label in labels:
            images_from_label = np.random.choice(
                self.images_per_label[label],
                self.n_support+self.n_query,
                replace=False
            )
            images_indices.extend(images_from_label)
        return torch.tensor(images_indices, dtype=torch.int32)

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            labels = self.sample_labels()
            yield self.sample_images_from_labels(labels)
