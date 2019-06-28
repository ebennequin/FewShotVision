# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import numpy as np
import torch
import torch.nn as nn

from src.yolov3.utils.utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class


class YOLOMAML(nn.Module):
    def __init__(self,
                 base_model,
                 n_way,
                 n_support,
                 n_query,
                 image_size,
                 approx=True,
                 n_task=4,
                 task_update_num=5,
                 train_lr=0.01,
                 device='cpu',
                 ):
        '''

        Args:
            base_model (nn.Module): base neural network
            n_way (int): number of different classes
            n_support (int): number of examples per class in the support set
            n_query (int): number of examples per class in the query set
            image_size (int): size of images
            approx (bool): whether to use an approximation of the meta-backpropagation
            n_task (int): number of episodes between each meta-backpropagation
            task_update_num (int): number of updates inside each episode
            train_lr (float): learning rate for intra-task updates
            device (str): cuda or cpu
        '''
        super(YOLOMAML, self).__init__()

        self.loss_fn = lambda loss, dummy: loss

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.base_model = base_model
        self.image_size = image_size

        self.n_task = n_task
        self.task_update_num = task_update_num
        self.train_lr = train_lr
        self.approx = approx

        self.device = device

        self.to(self.device)

    def forward(self, x, targets):
        '''
        Computes the classification prediction for input data. If targets is None, the loss will not be part of
        the output.
        Args:
            x (torch.Tensor): shape (number_of_images, dim_of_images) input data
            targets (torch.Tensor): shape (number_of_boxes_in_all_images, 6) target boxes

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: respectively the YOLO output of shape (number_of_images,
            number_of_yolo_output_boxes, 5+n_way), and the loss resulting from this output, of shape 0
        '''

        return self.base_model.forward(x, targets)

    def set_forward(self, support_set, support_set_targets, query_set, query_set_targets=None):
        '''
        Fine-tunes parameters on support set and apply updated parameters on query set. If query_set_targets is None,
        the loss will not be part of the output.
        Args:
            support_set (torch.Tensor): shape (n_way*n_support, dim_of_img) support set images
            support_set_targets (torch.Tensor): shape (L, 6) where L is the sum of the number of boxes in support
            set images
            query_set (torch.Tensor): shape (n_way*n_query, dim_of_img) query set images
            query_set_targets (torch.Tensor): shape (L, 6) where L is the sum of the number of boxes in query
            set images

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: respectively the YOLO output of shape (n_way*n_query,
            number_of_yolo_output_boxes, 5+n_way), and the loss resulting from this output, of shape 0
        '''
        fast_parameters = [param for param in self.parameters() if param.requires_grad]

        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            support_set_loss, support_set_output = self.forward(support_set, support_set_targets)
            grad = torch.autograd.grad(support_set_loss, fast_parameters,
                                       create_graph=True)  # build full graph support gradient of gradient
            if self.approx:
                grad = [g.detach() for g in
                        grad]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            count = 0
            for weight in self.parameters():
                if weight.requires_grad:
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbones.py
                    if weight.fast is None:
                        weight.fast = weight - self.train_lr * grad[count]  # create weight.fast
                    else:
                        weight.fast = weight.fast - self.train_lr * grad[
                            count]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                    fast_parameters.append(
                        weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                    count += 1

        return self.forward(query_set, query_set_targets)

    def set_forward_loss(self, support_set, support_set_targets, query_set, query_set_targets):
        '''
        Computes the meta-training loss for one episode
        Args:
            support_set (torch.Tensor): shape (n_way*n_support, dim_of_img) support set images
            support_set_targets (torch.Tensor): shape (L, 6) where L is the sum of the number of boxes in support
            set images
            query_set (torch.Tensor): shape (n_way*n_query, dim_of_img) query set images
            query_set_targets (torch.Tensor): shape (L, 6) where L is the sum of the number of boxes in query
            set images

        Returns:
            torch.Tensor: shape 0, the loss of the model on this episode
        '''
        query_set_loss, query_set_output = self.set_forward(support_set, support_set_targets, query_set, query_set_targets)
        loss = self.loss_fn(query_set_loss, query_set_output)

        return loss

    def train_loop(self, epoch, train_loader, optimizer):
        '''
        Executes one meta-training epoch.
        Args:
            epoch (int): current epoch
            train_loader (DataLoader): loader of a given number of episodes
            optimizer (torch.optim.Optimizer): model optimizer

        '''
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        for episode_index, (paths, images, targets, labels) in enumerate(train_loader):
            targets = self.rename_labels(targets)
            support_set, support_set_targets, query_set, query_set_targets = self.split_support_and_query_set(
                images,
                targets
            )

            loss = self.set_forward_loss(support_set, support_set_targets, query_set, query_set_targets)
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
            if episode_index % print_freq == 0:
                print(
                    'Epoch {epoch} | Episode {episode}/{total_episodes} | Loss {loss}'.format(
                        epoch=epoch,
                        episode=episode_index,
                        total_episodes=len(train_loader),
                        loss=avg_loss/float(episode_index + 1)
                    )
                )

    def eval_loop(self, data_loader):
        '''
        Evaluates the model on detection tasks sampled by data_loader
        Args:
            data_loader (torch.utils.data.DataLoader): episodic detection data loader

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: respectively
            precision, recall, average precision, F1 score and per-class average precision of the model
        '''
        self.eval()

        batch_statistics = []
        labels = []

        for batch_index, (paths, images, targets, labels) in enumerate(data_loader):
            targets = self.rename_labels(targets)
            support_set, support_set_targets, query_set, query_set_targets = self.split_support_and_query_set(
                images,
                targets
            )

            outputs_on_query = self.set_forward(support_set, support_set_targets, query_set)
            outputs_on_query = non_max_suppression(outputs_on_query)

            query_set_targets[:, 2:] = xywh2xyxy(query_set_targets[:, 2:])
            query_set_targets[:, 2:] *= self.image_size

            batch_statistics += get_batch_statistics(outputs_on_query, query_set_targets, 0.8)

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*batch_statistics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        return precision, recall, AP, f1, ap_class

    def rename_labels(self, targets):
        '''

        Args:
            targets (torch.Tensor): targets given by the data loader

        Returns:
            torch.Tensor: same targets but the labels all lie in range(n_way)
        '''
        old_labels = np.unique(targets[:, 1])
        labels_mapping = {}
        for new_label, old_label in enumerate(old_labels):
            labels_mapping[old_label] = new_label
        for box in targets:
            box[1] = labels_mapping[float(box[1])]

        return targets

    def split_support_and_query_set(self, images, targets):
        '''
        Split images and targets between support set and query set
        Args:
            images (torch.Tensor): shape (n_way*(n_support+n_query), dim_of_img)
            targets (torch.Tensor): shape (L, 6) where L is the sum of the number of boxes in every images

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: both images and targets, split between
            support set and query set
        '''
        # Split images between support set and query set
        support_set_list = []
        query_set_list = []
        support_indices = []
        query_indices = []
        for index, image in enumerate(images):
            if index % (self.n_support+self.n_query) < self.n_support:
                support_set_list.append(image.unsqueeze(0))
                support_indices.append(index)
            else:
                query_set_list.append(image.unsqueeze(0))
                query_indices.append(index)

        support_set = torch.cat(support_set_list)
        query_set = torch.cat(query_set_list)

        # Split targets between support set and query set
        support_targets_list = []
        query_targets_list = []
        for box in targets:
            if int(box[0]) in support_indices:
                support_targets_list.append(box.unsqueeze(0))
            else:
                query_targets_list.append(box.unsqueeze(0))

        support_targets = torch.cat(support_targets_list)
        query_targets = torch.cat(query_targets_list)

        # The first element of each line of targets refers to the image associated with the box.
        # This ensures this first element lies between 0 and the size of each splitted set of images
        for x_targets in [support_targets, query_targets]:
            old_indices = np.unique(x_targets[:,0])
            indices_mapping = {}
            for new_index, old_index in enumerate(old_indices):
                indices_mapping[old_index] = new_index
            for box in x_targets:
                box[0] = indices_mapping[float(box[0])]

        return support_set.to(self.device), support_targets.to(self.device), query_set.to(self.device), query_targets.to(self.device)
