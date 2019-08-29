import os


import torch

from classification.src.loaders.data_managers import SimpleDataManager, SetDataManager
from classification.src.methods import BaselineTrain
from classification.src.methods import ProtoNet
from classification.src.methods import MatchingNet
from classification.src.methods import RelationNet
from classification.src.methods import MAML
from utils import configs, backbones
from utils.io_utils import (
    model_dict,
    path_to_step_output,
    set_and_print_random_seed,
    get_path_to_json,
)


class MethodTraining():
    """
    This step handles the training of the algorithm on the base dataset
    """

    def __init__(
            self,
            dataset,
            backbone='Conv4',
            method='baseline',
            train_n_way=5,
            test_n_way=5,
            n_shot=5,
            train_aug=False,
            shallow=False,
            num_classes=4412,
            start_epoch=0,
            stop_epoch=-1,
            resume=False,
            warmup=False,
            optimizer='Adam',
            learning_rate=0.001,
            n_episode=100,
            random_seed=None,
            output_dir=configs.save_dir,
            n_swaps=0,
    ):
        """
        Args:
            dataset (str): CUB/miniImageNet/cross/omniglot/cross_char
            backbone (str): Conv{4|6} / ResNet{10|18|34|50|101}
            method (str): baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}
            train_n_way (int): number of labels in a classification task during training
            test_n_way (int): number of labels in a classification task during testing
            n_shot (int): number of labeled data in each class
            train_aug (bool): perform data augmentation or not during training
            shallow (bool): reduces the dataset to 256 images (typically for quick code testing)
            num_classes (int): total number of classes in softmax, only used in baseline #TODO delete this parameter
            start_epoch (int): starting epoch
            stop_epoch (int): stopping epoch
            resume (bool): continue from previous trained model with largest epoch
            warmup (bool): continue from baseline, neglected if resume is true
            optimizer (str): must be a valid class of torch.optim (Adam, SGD, ...)
            learning_rate (float): learning rate fed to the optimizer
            n_episode (int): number of episodes per epoch during meta-training
            random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
            output_dir (str): path to experiments output directory
            n_swaps (int): number of swaps between labels in the support set of each episode, in order to
            test the robustness to label noise
        """

        self.dataset = dataset
        self.backbone = backbone
        self.method = method
        self.train_n_way = train_n_way
        self.test_n_way = test_n_way
        self.n_shot = n_shot
        self.train_aug = train_aug
        self.shallow = shallow
        self.num_classes = num_classes
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.resume = resume
        self.warmup = warmup
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_episode = n_episode
        self.random_seed = random_seed
        self.n_swaps = n_swaps

        if self.dataset in ['omniglot', 'cross_char']:
            assert self.backbone == 'Conv4' and not self.train_aug, 'omniglot only support Conv4 without augmentation'
            self.backbone = 'Conv4S'

        self.checkpoint_dir = path_to_step_output(
            self.dataset,
            self.backbone,
            self.method,
            output_dir
        )


    def apply(self):
        """
        Execute the MethodTraining step
        Returns:
            dict: a dictionary containing the whole state of the model that gave the higher validation accuracy

        """
        set_and_print_random_seed(self.random_seed, True, self.checkpoint_dir)

        base_loader, val_loader, model = self._get_data_loaders_and_model()

        return self._train(base_loader, val_loader, model)

    def dump_output(self, _, output_folder, output_name, **__):
        pass

    def _train(self, base_loader, val_loader, model):
        """
        Trains the model on the base set
        Args:
            base_loader (torch.utils.data.DataLoader): data loader for base set
            val_loader (torch.utils.data.DataLoader): data loader for validation set
            model (torch.nn.Module): neural network model to train

        Returns:
            dict: a dictionary containing the whole state of the model that gave the higher validation accuracy

        """
        optimizer = self._get_optimizer(model)
        max_acc = 0
        best_model_epoch = -1
        best_model_state = model.state_dict()

        for epoch in range(self.start_epoch, self.stop_epoch):
            model.train()
            model.train_loop(epoch, base_loader, optimizer, self.n_swaps)  # model are called by reference, no need to return
            model.eval()

            acc = model.eval_loop(val_loader, self.n_swaps)
            # TODO: check that it makes sense to train baselines systematically for 400 epochs (and not validate)
            if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(self.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
                best_model_epoch = epoch
                best_model_state = model.state_dict()

            if epoch == self.stop_epoch - 1:
                outfile = os.path.join(self.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        return {'epoch': best_model_epoch, 'state': best_model_state}

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

    def _get_data_loaders_and_model(self):
        """ Function that returns train/val data loaders and the model

        Returns:
            tuple: a tuple of 3 elements containing the train/val data loaders and the model
        """
        path_to_base_file = get_path_to_json(self.dataset, 'base')
        path_to_val_file = get_path_to_json(self.dataset, 'val')

        # Define size of input image depending on backbone and dataset
        if 'Conv' in self.backbone:
            if self.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84
        else:
            image_size = 224

        self._set_default_epochs()

        # Define data loaders and model
        if self.method in ['baseline', 'baseline++']:
            base_data_manager = SimpleDataManager(image_size, batch_size=16)
            base_loader = base_data_manager.get_data_loader(path_to_base_file, aug=self.train_aug, shallow=self.shallow)
            val_data_manager = SimpleDataManager(image_size, batch_size=64)
            val_loader = val_data_manager.get_data_loader(path_to_val_file, aug=False)

            if self.dataset == 'omniglot':
                # TODO : change num_classes
                assert self.num_classes >= 4112, 'class number need to be larger than max label id in base class'
            if self.dataset == 'cross_char':
                assert self.num_classes >= 1597, 'class number need to be larger than max label id in base class'

            if self.method == 'baseline':
                model = BaselineTrain(model_dict[self.backbone], self.num_classes)
            elif self.method == 'baseline++':
                model = BaselineTrain(model_dict[self.backbone], self.num_classes, loss_type='dist')

        elif self.method in ['protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
            n_query = max(1, int(
                16 * self.test_n_way / self.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

            train_few_shot_params = dict(n_way=self.train_n_way, n_support=self.n_shot)
            base_data_manager = SetDataManager(
                image_size,
                n_query=n_query,
                n_episode=self.n_episode,
                **train_few_shot_params,
            )
            base_loader = base_data_manager.get_data_loader(path_to_base_file, aug=self.train_aug)

            test_few_shot_params = dict(n_way=self.test_n_way, n_support=self.n_shot)
            val_data_manager = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
            val_loader = val_data_manager.get_data_loader(path_to_val_file, aug=False)
            # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

            if self.method == 'protonet':
                model = ProtoNet(model_dict[self.backbone], **train_few_shot_params)
            elif self.method == 'matchingnet':
                model = MatchingNet(model_dict[self.backbone], **train_few_shot_params)
            elif self.method in ['relationnet', 'relationnet_softmax']:
                if self.backbone == 'Conv4':
                    feature_model = backbones.Conv4NP
                elif self.backbone == 'Conv6':
                    feature_model = backbones.Conv6NP
                elif self.backbone == 'Conv4S':
                    feature_model = backbones.Conv4SNP
                else:
                    feature_model = lambda: model_dict[self.backbone](flatten=False)
                loss_type = 'mse' if self.method == 'relationnet' else 'softmax'

                model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
            elif self.method in ['maml', 'maml_approx']:
                backbones.ConvBlock.maml = True
                backbones.SimpleBlock.maml = True
                backbones.BottleneckBlock.maml = True
                backbones.ResNet.maml = True
                model = MAML(model_dict[self.backbone], approx=(self.method == 'maml_approx'), **train_few_shot_params)
                if self.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                    model.n_task = 32
                    model.task_update_num = 1
                    model.train_lr = 0.1
        else:
            raise ValueError('Unknown method')

        model = model.cuda()


        if self.method == 'maml' or self.method == 'maml_approx':
            self.stop_epoch = self.stop_epoch * model.n_task  # maml use multiple tasks in one update

        return (
            base_loader,
            val_loader,
            model,
        )

    def _set_default_epochs(self):
        """
        Defines the number of epoch if stop_epoch has been initialized to -1, with arbitrary values depending
        on the method and dataset
        """
        if self.stop_epoch == -1:
            if self.method in ['baseline', 'baseline++']:
                if self.dataset in ['omniglot', 'cross_char']:
                    self.stop_epoch = 5
                elif self.dataset in ['CUB']:
                    self.stop_epoch = 200
                elif self.dataset in ['miniImageNet', 'cross']:
                    self.stop_epoch = 400
                else:
                    self.stop_epoch = 400
            else:
                if self.n_shot == 1:
                    self.stop_epoch = 600
                elif self.n_shot == 5:
                    self.stop_epoch = 400
                else:
                    self.stop_epoch = 600
