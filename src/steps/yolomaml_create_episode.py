import os

import numpy as np
from pipeline.steps import AbstractStep
import pickle

from src.utils import configs

from src.loaders.dataset import DetectionTaskSampler
from src.yolov3.utils.datasets import ListDataset
from src.yolov3.utils.parse_config import parse_data_config


class YOLOMAMLCreateEpisode(AbstractStep):
    """
    This step creates a detection task. It will output :
        - a .txt file containing the paths to the images composing the episode
        - a .data file containing the episode configuration
    """

    def __init__(
            self,
            dataset_config,
            n_way,
            n_shot,
            n_query,
            output_dir='./data/coco/episodes',
            episode_name=None,
            labels=None,
    ):
        """

        Args:
            dataset_config (str): path to data config file
            n_way (int): number of different classes in the episode
            n_shot (int): number of support set instances per class
            n_query (int): number of query set instances per class
            output_dir (str): output directory
            episode_name (str): name of the output file (without type extension)
            labels (list): labels from which to sample instances. If None, labels will be sampled at random.
        """

        self.data_config = parse_data_config(dataset_config)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.labels = labels

        self.episode_name = episode_name
        self.output_dir = output_dir

    def apply(self):
        """
        Execute the YOLOMAMLCreateEpisode step

        """

        dataset = ListDataset(self.data_config['eval'])

        sampler = DetectionTaskSampler(
            dataset,
            self.n_way,
            self.n_shot,
            self.n_query,
            1,
            self.data_config['eval_dict_path'],
        )

        if self.labels is None:
            self.labels = list(sampler.sample_labels())
        else:
            if len(self.labels) != self.n_way:
                raise ValueError('You have to provide exactly n_way labels')
            if not all(label in sampler.label_list for label in self.labels):
                raise ValueError("At least one label doesn't have enough instances in the dataset")

        image_indices = list(sampler.sample_images_from_labels(np.array(self.labels))[self.n_way:])

        images_path = []

        for index in image_indices:
            images_path.append(dataset[index][0])

        output_name = self.get_output_name()

        with open(output_name + '.txt', 'w') as f:
            for path in images_path:
                f.write(path + '\n')

        with open(output_name + '.data', 'w') as f:
            f.write('labels='+str(self.labels).strip('[]')+'\n')
            f.write('eval='+output_name+'.txt'+'\n')
            f.write('classes='+str(self.data_config['classes'])+'\n')
            f.write('names='+str(self.data_config['names'])+'\n')

    def dump_output(self, _, output_folder, output_name, **__):
        pass

    def get_output_name(self):
        if self.episode_name is None:
            self.episode_name = 'task'
            for label in self.labels:
                self.episode_name = self.episode_name + '-' + str(label)

        return os.path.join(self.output_dir, self.episode_name)
