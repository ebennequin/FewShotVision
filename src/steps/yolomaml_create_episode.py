import os

import numpy as np
from pipeline.steps import AbstractStep
import pickle

from src.utils import configs

from src.loaders.dataset import create_dict_images_per_label
from src.yolov3.utils.datasets import ListDataset
from src.yolov3.utils.parse_config import parse_data_config


class YOLOMAMLCreateEpisode(AbstractStep):
    """
    This step creates the dictionary requires for yolo MAML
    """

    def __init__(
            self,
            dataset_config,
            n_way,
            n_shot,
            n_query,
            labels=None,
    ):

        self.data_config = parse_data_config(dataset_config)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.labels = labels if labels is not None else self.get_random_labels()

    def apply(self):
        """
        Execute the YOLOMAMLCreateEpisode step

        """

        with open(self.data_config['valid_dict'], 'rb') as dictionary_file:
            images_per_label = pickle.load(dictionary_file)

        images_path = []

        for label in self.labels:
            images_from_label = np.random.choice(
                self.images_per_label[label],
                self.n_support+self.n_query,
                replace=False
            )
            images_path.extend(images_from_label)

    def dump_output(self, _, output_folder, output_name, **__):
        pass

    def get_random_labels(self):
        labels_number = self.data_config['classes']
        labels = np.sort(np.random.choice(labels_number, self.n_way, replace=False))

        return labels
