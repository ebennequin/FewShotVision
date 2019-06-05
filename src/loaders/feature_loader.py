import torch
import numpy as np
import h5py


class SimpleHDF5Dataset:
    def __init__(self, file_handle=None):
        if file_handle == None:
            self.f = ''
            self.all_features_dset = []
            self.all_labels = []
            self.total = 0
        else:
            self.f = file_handle
            self.all_features_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            self.total = self.f['count'][0]

    def __getitem__(self, i):
        return torch.Tensor(self.all_features_dset[i, :]), int(self.all_labels[i])

    def __len__(self):
        return self.total


def init_loader(filename, features_and_labels=None):
    '''

    Args:
        filename (str): path to .h5py file containing the features and labels
        features_and_labels (tuple): a tuple (features, labels). if None, loads from .h5py file

    Returns:
        dict: a dict where keys are the labels and values are the corresponding feature vectors
    '''
    if features_and_labels is None:
        with h5py.File(filename, 'r') as f:
            fileset = SimpleHDF5Dataset(f)
        features = fileset.all_features_dset
        labels = fileset.all_labels
    else:
        features, labels = features_and_labels

    while not features[-1].any():
        features = np.delete(features, -1, axis=0)
        labels = np.delete(labels, -1, axis=0)

    features_per_label = {
        label: []
        for label in np.unique(np.array(labels)).tolist()
    }

    for ind in range(len(labels)):
        features_per_label[labels[ind]].append(features[ind])

    return features_per_label
