import os

import numpy as np
from pipeline.steps import AbstractStep
import pickle

from src.utils import configs

from src.loaders.dataset import DetectionTaskSampler
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
        self.episode_config = episode_config
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


    def apply(self):
        pass

    def dump_output(self, _, output_folder, output_name, **__):
        pass
