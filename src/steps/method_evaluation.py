import os
import random
import time

import numpy as np
from pipeline.steps import AbstractStep
import torch
import torch.optim
import torch.utils.data.sampler

from src import backbone
import src.loaders.feature_loader as feat_loader  # TODO : ambiguous
from src.loaders.datamgr import SetDataManager
from src.methods import BaselineFinetune
from src.methods import ProtoNet
from src.methods import MatchingNet
from src.methods import RelationNet
from src.methods.maml import MAML
from src.utils import configs
from src.utils.io_utils import model_dict, get_best_file, get_assigned_file, path_to_step_output


class MethodEvaluation(AbstractStep):
    '''
    This step handles the evaluation of the trained model on the novel dataset
    '''
    def __init__(
            self,
            dataset,
            backbone='Conv4',
            method='baseline',
            train_n_way=5,
            test_n_way=5,
            n_shot=5,
            train_aug=False,
            split='novel',
            save_iter=-1,
            n_iter=600,
            adaptation=False,
    ):
        '''
        Args:
            dataset (str): CUB/miniImageNet/cross/omniglot/cross_char
            model (str): Conv{4|6} / ResNet{10|18|34|50|101}
            method (str): baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}
            train_n_way (int): number of labels in a classification task during training
            test_n_way (int): number of labels in a classification task during testing
            n_shot (int): number of labeled data in each class
            train_aug (bool): perform data augmentation or not during training
            split (str): which dataset is considered (base, val or novel)
            save_iter (int): save feature from the model trained in x epoch, use the best model if x is -1
            n_iter (int): number of classification tasks on which the model is tested
            adaptation (boolean): further adaptation in test time or not
        '''
        self.dataset = dataset
        self.backbone = backbone
        self.method = method
        self.train_n_way = train_n_way
        self.test_n_way = test_n_way
        self.n_shot = n_shot
        self.train_aug = train_aug
        self.split = split
        self.save_iter = save_iter
        self.n_iter = n_iter
        self.adaptation = adaptation

        if self.dataset in ['omniglot', 'cross_char']:
            assert self.backbone == 'Conv4' and not self.train_aug, 'omniglot only support Conv4 without augmentation'
            self.backbone = 'Conv4S'

        self.checkpoint_dir = path_to_step_output(
            self.dataset,
            self.backbone,
            self.method,
            self.train_n_way,
            self.n_shot,
            self.train_aug,
        )


    def apply(self, model_state=None, features_and_labels=None):

        acc_all = []

        model = self._load_model(model_state)

        split = self.split
        if self.save_iter != -1:
            split_str = split + "_" + str(self.save_iter)
        else:
            split_str = split
        if self.method in ['maml', 'maml_approx']:  # maml do not support testing with feature
            if 'Conv' in self.backbone:
                if self.dataset in ['omniglot', 'cross_char']:
                    image_size = 28
                else:
                    image_size = 84
            else:
                image_size = 224

            datamgr = SetDataManager(image_size, n_episode=self.n_iter, n_query=15, **few_shot_params)

            if self.dataset == 'cross':
                if split == 'base':
                    loadfile = configs.data_dir['miniImageNet'] + 'all.json'
                else:
                    loadfile = configs.data_dir['CUB'] + split + '.json'
            elif self.dataset == 'cross_char':
                if split == 'base':
                    loadfile = configs.data_dir['omniglot'] + 'noLatin.json'
                else:
                    loadfile = configs.data_dir['emnist'] + split + '.json'
            else:
                loadfile = configs.data_dir[self.dataset] + split + '.json'

            novel_loader = datamgr.get_data_loader(loadfile, aug=False)
            if self.adaptation:
                model.task_update_num = 100  # We perform adaptation on MAML simply by updating more times.
            model.eval()
            acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)

        else:
            # Fetch feature vectors
            # cl_data_file is a dictionnary where each key is a label and each value is a list of feature vectors
            novel_file = os.path.join(self.checkpoint_dir,
                                      split_str + ".hdf5")  # defaut split = novel, but you can also test base or val classes
            cl_data_file = feat_loader.init_loader(novel_file, features_and_labels)

            for i in range(self.n_iter):
                acc = self._feature_evaluation(cl_data_file, model, n_query=15)
                acc_all.append(acc)
                if i % 10 == 0:
                    print('{}/{}'.format(i, self.n_iter))

            acc_all = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std = np.std(acc_all)
            print('%d Test Acc = %4.2f%% +- %4.2f%%' % (self.n_iter, acc_mean, 1.96 * acc_std / np.sqrt(
                self.n_iter)))  # 1.96 is the approximation for 95% confidence interval
        with open(os.path.join(self.checkpoint_dir, 'results.txt'), 'w') as f:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            aug_str = '-aug' if self.train_aug else ''
            aug_str += '-adapted' if self.adaptation else ''
            if self.method in ['baseline', 'baseline++']:
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' % (
                    self.dataset, split_str, self.backbone, self.method, aug_str, self.n_shot, self.test_n_way)
            else:
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' % (
                    self.dataset, split_str, self.backbone, self.method, aug_str, self.n_shot, self.train_n_way,
                    self.test_n_way)
            acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' % (
                self.n_iter, acc_mean, 1.96 * acc_std / np.sqrt(self.n_iter))  # TODO : redite
            f.write('Time: %s, Setting: %s, Acc: %s \n' % (timestamp, exp_setting, acc_str))

    def dump_output(self, _, output_folder, output_name, **__):
        pass

    def _feature_evaluation(self, cl_data_file, model, n_query=15):
        class_list = cl_data_file.keys()

        select_class = random.sample(class_list, self.test_n_way)
        z_all = []
        for cl in select_class:
            img_feat = cl_data_file[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(self.n_shot + n_query)])  # stack each batch

        z_all = torch.from_numpy(np.array(z_all))

        model.n_query = n_query
        if self.adaptation:
            scores = model.set_forward_adaptation(z_all, is_feature=True)
        else:
            scores = model.set_forward(z_all, is_feature=True)
        pred = scores.data.cpu().numpy().argmax(axis=1)
        y = np.repeat(range(self.test_n_way), n_query)
        acc = np.mean(pred == y) * 100
        return acc

    def _load_model(self, model_state=None):
        '''
        Load model from training
        Args:
            model_state: dict containing the state of the trained model. If None, loads from .tar file

        Returns:
            model: torch module
        '''
        few_shot_params = dict(n_way=self.test_n_way, n_support=self.n_shot)

        # Define model
        if self.method == 'baseline':
            model = BaselineFinetune(model_dict[self.backbone], **few_shot_params)
        elif self.method == 'baseline++':
            model = BaselineFinetune(model_dict[self.backbone], loss_type='dist', **few_shot_params)
        elif self.method == 'protonet':
            model = ProtoNet(model_dict[self.backbone], **few_shot_params)
        elif self.method == 'matchingnet':
            model = MatchingNet(model_dict[self.backbone], **few_shot_params)
        elif self.method in ['relationnet', 'relationnet_softmax']:
            if self.backbone == 'Conv4':
                feature_model = backbone.Conv4NP
            elif self.backbone == 'Conv6':
                feature_model = backbone.Conv6NP
            elif self.backbone == 'Conv4S':
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[self.backbone](flatten=False)
            loss_type = 'mse' if self.method == 'relationnet' else 'softmax'
            model = RelationNet(feature_model, loss_type=loss_type, **few_shot_params)
        elif self.method in ['maml', 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = MAML(model_dict[self.backbone], approx=(self.method == 'maml_approx'), **few_shot_params)
            if self.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
        else:
            raise ValueError('Unknown method')

        model = model.cuda()

        # Fetch model parameters
        if not self.method in ['baseline', 'baseline++']:
            if model_state == None:
                if self.save_iter != -1:
                    modelfile = get_assigned_file(self.checkpoint_dir, self.save_iter)
                else:
                    modelfile = get_best_file(self.checkpoint_dir)
                if modelfile is not None:
                    tmp = torch.load(modelfile)
                    model.load_state_dict(tmp['state'])
            else:
                model.load_state_dict(model_state['state'])

        return model
