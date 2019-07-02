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

def load_features_and_labels_from_file(filename):
    """

    Args:
        filename (str): path to .h5py file containing the features and labels

    Returns:
        array: extracted features
        array: extracted labels
    """
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f)
    features = fileset.all_features_dset
    labels = fileset.all_labels
    return features, labels
