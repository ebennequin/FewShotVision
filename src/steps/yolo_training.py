import os

from pipeline.steps import AbstractStep
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from src.loaders.data_managers import DetectionSetDataManager
from src.methods import YOLOMAML
from src.utils import configs
from src.utils.utils import include_episode_loss_dict
from src.utils.io_utils import set_and_print_random_seed
from src.yolov3.model import Darknet
from src.yolov3.utils.datasets import ListDataset
from src.yolov3.utils.parse_config import parse_data_config


class YOLOTraining(AbstractStep):
    """
    This step handles the training of the algorithm on the base dataset
    """

    def __init__(
            self,
            dataset_config='yolov3/config/black.data',
            model_config='yolov3/config/yolov3.cfg',
            pretrained_weights=None,
            optimizer='Adam',
            learning_rate=0.001,
            multiscale_training=True,
            batch_size=32,
            n_cpu=8,
            gradient_accumulation=10,
            print_freq=1,
            validation_freq=5,
            n_epoch=100,
            objectness_threshold=0.8,
            nms_threshold=0.4,
            iou_threshold=0.2,
            image_size=416,
            random_seed=None,
            output_dir=configs.save_dir,
    ):
        """
        Args:
            dataset_config (str): path to data config file
            model_config (str): path to model definition file
            pretrained_weights (str): path to a file containing pretrained weights for the model
            optimizer (str): must be a valid class of torch.optim (Adam, SGD, ...)
            learning_rate (float): learning rate fed to the optimizer
            multiscale_training (bool): whether to sample batches with different image sizes
            batch_size (int): size of a training batch
            n_cpu (int): number of workers for the computation of the dataloader
            gradient_accumulation (int): number of gradients from batches to accumulate before a gradient descent
            print_freq (int): inside an epoch, print status update every print_freq episodes
            validation_freq (int): inside an epoch, frequency with which we evaluate the model on the validation set
            n_epoch (int): number of meta-training epochs
            objectness_threshold (float): at evaluation time, only keep boxes with objectness above this threshold
            nms_threshold (float): threshold for non maximum suppression, at evaluation time
            iou_threshold (float): threshold for intersection over union
            image_size (int): size of images (square)
            random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
            output_dir (str): path to experiments output directory
        """

        self.dataset_config = dataset_config
        self.model_config = model_config
        self.pretrained_weights = pretrained_weights
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.multiscale_training = multiscale_training
        self.batch_size = batch_size
        self.n_cpu = n_cpu
        self.gradient_accumulation = gradient_accumulation
        self.print_freq = print_freq
        self.validation_freq = validation_freq
        self.n_epoch = n_epoch
        self.objectness_threshold = objectness_threshold
        self.nms_threshold = nms_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.random_seed = random_seed
        self.checkpoint_dir = output_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.writer = SummaryWriter(log_dir=output_dir)

    def apply(self):
        """
        Execute the YOLOMAMLTraining step
        Returns:
            dict: a dictionary containing the whole state of the model that gave the higher validation accuracy

        """
        set_and_print_random_seed(self.random_seed, True, self.checkpoint_dir)

        data_config = parse_data_config(self.dataset_config)
        train_path = data_config["train"]
        valid_path = data_config.get("valid", None)

        train_loader = self._get_data_loader(train_path)
        val_loader = self._get_data_loader(valid_path)

        model = self._get_model()

        return self._train(train_loader, val_loader, model)

    def dump_output(self, _, output_folder, output_name, **__):
        pass

    def _train(self, train_loader, val_loader, model):
        """
        Trains the model on the training set
        Args:
            train_loader (torch.utils.data.DataLoader): data loader for training set
            val_loader (torch.utils.data.DataLoader): data loader for validation set
            model (Darknet): neural network model to train

        Returns:
            dict: a dictionary containing the whole state of the model that gave the higher validation accuracy

        """
        optimizer = self._get_optimizer(model)
        optimizer.zero_grad()

        for epoch in range(self.n_epoch):
            loss_dict = {}

            model.train()
            for batch_index, (_, images, targets) in enumerate(train_loader):
                batch_loss_dict, _ = model.forward(images.to(self.device), targets.to(self.device))
                loss = batch_loss_dict['total_loss']
                loss.backward()
                loss_dict = include_episode_loss_dict(loss_dict, batch_loss_dict, len(train_loader))

                if batch_index % self.gradient_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            self.plot_tensorboard(loss_dict, epoch)

            if epoch % self.print_freq == 0:
                print(
                    'Epoch {epoch}/{n_epochs} | Loss {loss}'.format(
                        epoch=epoch,
                        n_epochs=self.n_epoch,
                        loss=loss_dict['total_loss'],
                    )
                )

        self.writer.close()

        model.save_darknet_weights(os.path.join(self.checkpoint_dir, 'final.weights'))

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

    def _get_data_loader(self, path_to_data_file):
        """

        Args:
            path_to_data_file (str): path to file containing paths to images

        Returns:
            torch.utils.data.DataLoader: samples data in the shape of batches
        """
        dataset = ListDataset(path_to_data_file, augment=True, multiscale=self.multiscale_training)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        return dataloader

    def _get_model(self):
        """

        Returns:
            Darknet: YOLO model
        """

        model = Darknet(self.model_config, self.image_size, self.pretrained_weights).to(self.device)

        return model

    def plot_tensorboard(self, loss_dict, epoch):
        """
        Writes into summary the values present in loss_dict
        Args:
            loss_dict (dict): contains the different parts of the average loss on one epoch. Each key describes
            a part of the loss (ex: query_classification_loss) and each value is a 0-dim tensor. This dictionary is
            required to contain the keys 'support_total_loss' and 'query_total_loss' which contains respectively the
            total loss on the support set, and the total meta-loss on the query set
            epoch (int): global step value in the summary

        Returns:

        """
        for key, value in loss_dict.items():
            self.writer.add_scalar(key, value, epoch)

        return
