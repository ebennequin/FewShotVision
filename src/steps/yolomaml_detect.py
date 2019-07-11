import os

import numpy as np
from pipeline.steps import AbstractStep
import pickle
import torch

from src.utils import configs

from src.loaders.dataset import DetectionTaskSampler
from src.methods import YOLOMAML
from src.yolov3.model import Darknet
from src.yolov3.utils.datasets import ListDataset
from src.yolov3.utils.parse_config import parse_data_config


class YOLOMAMLDetect(AbstractStep):

    def __init__(
            self,
            episode_config,
            model_config,
            trained_weights,
            learning_rate,
            task_update_num,
            objectness_threshold=0.8,
            nms_threshold=0.4,
            iou_threshold=0.2,
            image_size=416,
            random_seed=None,
            output_dir=configs.save_dir,
    ):
        self.data_config = parse_data_config(episode_config)
        self.model_config = model_config
        self.trained_weights = trained_weights
        self.learning_rate = learning_rate
        self.task_update_num = task_update_num
        self.objectness_threshold = objectness_threshold
        self.nms_threshold = nms_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.random_seed = random_seed
        self.output_dir = output_dir

        self.labels = self.parse_labels(self.data_config['labels'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def apply(self):
        model = self.get_model()
        paths, images, targets = self.get_episode()

        support_set, support_targets, query_set, query_targets = model.split_support_and_query_set(images, targets)

        _, query_output = model.set_forward(support_set, support_targets, query_set)


    def dump_output(self, _, output_folder, output_name, **__):
        pass

    def parse_labels(self, labels_str):
        """
        Gets labels from a string
        Args:
            labels_str (str): string from the data config file describing the labels of the episode

        Returns:
            list: labels of the episode
        """
        labels_str_split = labels_str.split(', ')
        labels = [int(label) for label in labels_str_split]

        return labels


    def get_episode(self):
        """
        Returns:
            Tuple[Tuple, torch.Tensor, torch.Tensor]: the paths, images and target boxes of data instances composing
            the episode described in the data configuration file
        """
        dataset = ListDataset(
            list_path=self.data_config['eval'],
            img_size=self.image_size,
            augment=False,
            multiscale=False,
            normalized_labels=True,
        )

        data_instances = [dataset[-label-1] for label in self.labels]
        data_instances.extend([dataset[i] for i in range(len(dataset))])

        paths, images, targets, _ = dataset.collate_fn(data_instances)

        return paths, images, targets

    def get_model(self):
        """
        Returns:
            YOLOMAML: meta-model
        """

        base_model = Darknet(self.model_config, self.image_size, self.trained_weights)

        model = YOLOMAML(
            base_model,
            self.data_config['n_way'],
            self.data_config['n_shot'],
            self.data_config['n_query'],
            self.image_size,
            approx=True,
            task_update_num=self.task_update_num,
            train_lr=self.learning_rate,
            objectness_threshold=self.objectness_threshold,
            nms_threshold=self.nms_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
        )

        return model
