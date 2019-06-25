# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import numpy as np
import torch
import torch.nn as nn

from src import backbones
from src.methods.meta_template import MetaTemplate
from src.utils.utils import random_swap_tensor

class MAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, approx=False):
        '''

        Args:
            model_func (function in backbones.py): function that returns the backbone of the model
            n_way (int): number of different classes
            n_support (int): number of examples per class in the support set
            approx (bool): whether to use an approximation of the meta-backpropagation
        '''
        super(MAML, self).__init__(model_func, n_way, n_support, change_way=False)

        self.loss_fn = nn.CrossEntropyLoss() #TODO: should be customable
        #TODO: here it adds a classifier layer at the top of the feature extractor. Needs to change
        self.classifier = backbones.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)  # TODO why

        self.n_task = 4 #TODO should be customable
        self.task_update_num = 5 #TODO should be customable
        self.train_lr = 0.01 #TODO should be customable
        self.approx = approx  # first order approx.

    def forward(self, x):
        '''
        Computes the classification prediction for input data
        Args:
            x (torch.Tensor): shape (number_of_images, dim_of_images) input data

        Returns:
            torch.Tensor: shape (number_of_images, n_way) prediction
        '''
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def set_forward(self, x, is_feature=False):
        assert is_feature == False, 'MAML do not support fixed feature'
        x = x.cuda()
        support_set = x[:, :self.n_support, :, :, :].contiguous().view(self.n_way * self.n_support, *x.size()[2:])
        query_set = x[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query, *x.size()[2:])
        support_set_labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()

        fast_parameters = list(self.parameters())  # the first gradient calculated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            scores = self.forward(support_set)
            set_loss = self.loss_fn(scores, support_set_labels)
            grad = torch.autograd.grad(set_loss, fast_parameters,
                                       create_graph=True)  # build full graph support gradient of gradient
            if self.approx:
                grad = [g.detach() for g in
                        grad]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbones.py
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]  # create weight.fast
                else:
                    weight.fast = weight.fast - self.train_lr * grad[
                        k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(
                    weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

        scores = self.forward(query_set)
        return scores

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature=False)
        query_set_labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
        loss = self.loss_fn(scores, query_set_labels)

        return loss

    def train_loop(self, epoch, train_loader, optimizer, n_swaps):
        '''
        Executes one training epoch. This overwrites the parent function in MetaTemplate.
        Args:
            epoch (int): current epoch
            train_loader (DataLoader): loader of a given number of episodes
            optimizer (torch.optim.Optimizer): model optimizer
            n_swaps (int): number of swaps between labels in the support set of each episode, in order to
            test the robustness to label noise

        '''
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"
            x = random_swap_tensor(x, n_swaps, self.n_support)

            loss = self.set_forward_loss(x)
            avg_loss = avg_loss + loss.item()
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))

    def eval_loop(self, test_loader, n_swaps=0, return_std=False):  # overwrite parrent function
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"
            x = random_swap_tensor(x, n_swaps, self.n_support)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))) #TODO: already set elsewhere
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
