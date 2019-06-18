from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from src.utils.utils import random_swap_tensor


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way=True):
        '''

        Args:
            model_func (src.backbones object): backbone function
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            change_way (bool): allow n_way to be different in training and evaluation
            n_swaps (int): number of swaps at each episode during meta-training

        '''
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way  # some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x): # TODO: seems unused
        out = self.feature.forward(x)
        return out

    # is_features tells wether x is a feature vector (True) or rough image tensor (False)
    # parse_features returns the features vectors of the support and query sets
    def parse_feature(self, x, is_feature):
        x = x.cuda()
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)  # TODO : check that this means self.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    #TODO: is this always the same images in the episodes ?
    def train_loop(self, epoch, train_loader, optimizer, n_swaps):
        '''

        Args:
            epoch (int): current epoch
            train_loader (DataLoader): loader of a given number of episodes
            optimizer (torch.optim.Optimizer): model optimizer
            n_swaps (int): number of swaps between labels in the support set of each episode, in order to
            test the robustness to label noise

        '''
        print_freq = 10

        avg_loss = 0
        for episode_index, (episode, _) in enumerate(train_loader):
            self.n_query = episode.size(1) - self.n_support
            if self.change_way:
                self.n_way = episode.size(0)
            episode = random_swap_tensor(episode, n_swaps, self.n_support)
            optimizer.zero_grad()
            loss = self.set_forward_loss(episode)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            if episode_index % print_freq == 0:
                print('Epoch {epoch} | Batch {episode_index}/{n_batches} | Loss {loss}'.format(
                    epoch=epoch,
                    episode_index=episode_index,
                    n_batches=len(train_loader),
                    loss=avg_loss/float(episode_index + 1)
                ))

    def eval_loop(self, test_loader, n_swaps=0):
        '''

        Args:
            test_loader (DataLoader): loader of a given number of episodes
            n_swaps (int): number of swaps between labels in the support set of each episode, in order to
            test the robustness to label noise

        Returns:
            float: average accuracy on evaluation set
        '''
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            x = random_swap_tensor(x, n_swaps, self.n_support)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean

    def set_forward_adaptation(self, x,
                               is_feature=True):  # further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores
