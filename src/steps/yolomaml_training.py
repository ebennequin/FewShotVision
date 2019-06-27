import os

import numpy as np
from pipeline.steps import AbstractStep
import torch

from src import backbones
from src.loaders.data_managers import SimpleDataManager, SetDataManager
from src.methods import BaselineTrain
from src.methods import ProtoNet
from src.methods import MatchingNet
from src.methods import RelationNet
from src.methods.maml import MAML
from src.utils import configs
from src.utils.io_utils import (
    model_dict,
    path_to_step_output,
    set_and_print_random_seed,
    get_path_to_json,
)


class YOLOMAMLTraining(AbstractStep):
    '''
    This step handles the training of the algorithm on the base dataset
    '''

    def __init__(
            self,
            dataset='yolov3/config/coco.data',
            model_config='yolov3/config/yolov3.cfg',
            n_way=5,
            n_shot=5,
            optimizer='Adam',
            learning_rate=0.001,
            n_epoch=100,
            n_episode=100,
            random_seed=None,
            output_dir=configs.save_dir,
    ):
        '''
        Args:
            dataset (str): CUB/miniImageNet/cross/omniglot/cross_char
            model_config (str): path to model definition file
            n_way (int): number of labels in a detection task
            n_shot (int): number of labeled data in each class
            optimizer (str): must be a valid class of torch.optim (Adam, SGD, ...)
            learning_rate (float): learning rate fed to the optimizer
            n_epoch (int): number of meta-training epochs
            n_episode (int): number of episodes per epoch during meta-training
            random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
            output_dir (str): path to experiments output directory
        '''

        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.n_episode = n_episode
        self.random_seed = random_seed
        self.checkpoint_dir = output_dir


    def apply(self):
        '''
        Execute the YOLOMAMLTraining step
        Returns:
            dict: a dictionary containing the whole state of the model that gave the higher validation accuracy

        '''
        set_and_print_random_seed(self.random_seed, True, self.checkpoint_dir)

        base_loader = self._get_data_loader()
        val_loader = None

        model = self._get_model()

        return self._train(base_loader, val_loader, model)

    def dump_output(self, _, output_folder, output_name, **__):
        pass

    def _train(self, base_loader, val_loader, model):
        '''
        Trains the model on the base set
        Args:
            base_loader (torch.utils.data.DataLoader): data loader for base set
            val_loader (torch.utils.data.DataLoader): data loader for validation set
            model (torch.nn.Module): neural network model to train

        Returns:
            dict: a dictionary containing the whole state of the model that gave the higher validation accuracy

        '''
        optimizer = self._get_optimizer(model)

        for epoch in range(self.n_epoch):
            model.train()
            model.train_loop(epoch, base_loader, optimizer)

            #TODO : add validation

            if epoch == self.n_epoch - 1:
                outfile = os.path.join(self.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        return {'epoch': self.n_epoch, 'state': model.state_dict()}

    def _get_optimizer(self, model):
        """
        Get the optimizer from string self.optimizer
        Args:
            model (torch.nn.Module): the model to be trained

        Returns: a torch.optim.Optimizer object parameterized with model parameters

        """
        assert hasattr(torch.optim, self.optimizer), "The optimization method is not a torch.optim object"
        optimizer = getattr(torch.optim, self.optimizer)(model.parameters(), lr=self.learning_rate)

        return optimizer

    def _get_data_loader(self):
        pass

    def _get_model(self):
        pass
