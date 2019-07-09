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
                 task_update_num=5,
                 train_lr=0.01,
                 print_freq=10,
                 objectness_threshold=0.8,
                 nms_threshold=0.4,
                 iou_threshold=0.2,
                 device='cpu',
                 ):
        """

        Args:
            base_model (nn.Module): base neural network
            n_way (int): number of different classes
            n_support (int): number of examples per class in the support set
            n_query (int): number of examples per class in the query set
            image_size (int): size of images (square)
            approx (bool): whether to use an approximation of the meta-backpropagation
            task_update_num (int): number of updates inside each episode
            train_lr (float): learning rate for intra-task updates
            objectness_threshold (float): at evaluation time, only keep boxes with objectness above this threshold
            nms_threshold (float): threshold for non maximum suppression, at evaluation time
            iou_threshold (float): threshold for intersection over union
            print_freq (int): inside an epoch, print status update every print_freq episodes
            device (str): cuda or cpu
        """
        super(YOLOMAML, self).__init__()

        self.loss_fn = lambda loss, dummy: loss

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.base_model = base_model
        self.image_size = image_size

        self.task_update_num = task_update_num
        self.train_lr = train_lr
        self.approx = approx
        self.print_freq = print_freq

        self.objectness_threshold = objectness_threshold
        self.nms_threshold = nms_threshold
        self.iou_threshold = iou_threshold

        self.device = device

        self.to(self.device)

    def forward(self, x, targets):
        """
        Computes the classification prediction for input data.
        Args:
            x (torch.Tensor): shape (number_of_images, dim_of_images) input data
            targets (torch.Tensor): shape (number_of_boxes_in_all_images, 6) target boxes

        Returns:
            Tuple[dict, torch.Tensor]: respectively :
             - a dictionary containing the different parts of the loss resulting from the output. Each key describes
             a part of the loss (ex: classification_loss) and each value is a 0-dim tensor. This dictionary is
             required to contain the key 'total_loss' which contains the total loss resulting from the output.
             If targets is None, the dictionary will be empty.
             - the YOLO output of shape (number_of_images, number_of_yolo_output_boxes, 5+n_way). One line
            (of size 5+n_way) of the YOLO output contains 4 items about the box prediction, one about the objectness
            and n_way about the classification, in that order.
        """

        return self.base_model.forward(x, targets)

    def set_forward(self, support_set, support_set_targets, query_set, query_set_targets=None):
        """
        Fine-tunes parameters on support set and apply updated parameters on query set.
        Args:
            support_set (torch.Tensor): shape (n_way*n_support, dim_of_img) support set images
            support_set_targets (torch.Tensor): shape (L, 6) where L is the sum of the number of boxes in support
            set images
            query_set (torch.Tensor): shape (n_way*n_query, dim_of_img) query set images
            query_set_targets (torch.Tensor): shape (L, 6) where L is the sum of the number of boxes in query
            set images

        Returns:
            Tuple[dict, torch.Tensor]: respectively :
             - a dictionary containing the different parts of the loss resulting from this output. Each key describes
             a part of the loss (ex: query_classification_loss) and each value is a 0-dim tensor. This dictionary is
             required to contain the keys 'support_total_loss' and 'query_total_loss' which contains respectively the
             total loss on the support set, and the total meta-loss on the query set
             - the YOLO output of shape (number_of_images, number_of_yolo_output_boxes, 5+n_way). One line
            (of size 5+n_way) of the YOLO output contains 4 items about the box prediction, one about the objectness
            and n_way about the classification, in that order.
        """
        fast_parameters = [param for param in self.parameters() if param.requires_grad]

        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        support_loss_dict = {}

        for task_step in range(self.task_update_num):
            support_loss_dict, support_set_output = self.forward(support_set, support_set_targets)
            support_set_loss = support_loss_dict['total_loss']

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
                        weight.fast = weight.fast - self.train_lr * grad[count]
                    fast_parameters.append(weight.fast)
                    count += 1

        torch.cuda.empty_cache()

        query_loss_dict, query_output = self.forward(query_set, query_set_targets)

        complete_loss_dict = self.get_complete_loss_dict(support_loss_dict, query_loss_dict)

        return complete_loss_dict, query_output

    def set_forward_loss(self, support_set, support_set_targets, query_set, query_set_targets):
        """
        Computes the meta-training loss for one episode
        Args:
            support_set (torch.Tensor): shape (n_way*n_support, dim_of_img) support set images
            support_set_targets (torch.Tensor): shape (L, 6) where L is the sum of the number of boxes in support
            set images
            query_set (torch.Tensor): shape (n_way*n_query, dim_of_img) query set images
            query_set_targets (torch.Tensor): shape (L, 6) where L is the sum of the number of boxes in query
            set images

        Returns:
            dict: contains the different parts of the loss resulting from this output. Each key describes
            a part of the loss (ex: query_classification_loss) and each value is a 0-dim tensor. This dictionary is
            required to contain the keys 'support_total_loss' and 'query_total_loss' which contains respectively the
            total loss on the support set, and the total meta-loss on the query set
        """
        loss_dict, query_output = self.set_forward(
            support_set,
            support_set_targets,
            query_set,
            query_set_targets
        )

        return loss_dict

    def train_loop(self, epoch, train_loader, optimizer):
        """
        Executes one meta-training epoch. Executes several episodes then one meta-backpropagation.
        Args:
            epoch (int): current epoch
            train_loader (DataLoader): loader of a given number of episodes.  It returns a tuple of size 4 respectively
            containing the paths, the images, the targets and the labels
            optimizer (torch.optim.Optimizer): model optimizer

        Returns:
            float: average loss of the model on the query set of the episodes

        """
        self.train()

        cumulative_loss = 0
        loss_all = []
        optimizer.zero_grad()
        loss_dict = {}

        for episode_index, (paths, images, targets, labels) in enumerate(train_loader):
            targets = self.rename_labels(targets)
            support_set, support_set_targets, query_set, query_set_targets = self.split_support_and_query_set(
                images,
                targets
            )

            episode_loss_dict = self.set_forward_loss(support_set, support_set_targets, query_set, query_set_targets)
            loss = episode_loss_dict['query_total_loss']

            loss_dict = self.include_episode_loss_dict(loss_dict, episode_loss_dict, len(train_loader))
            cumulative_loss = cumulative_loss + loss.item()
            loss_all.append(loss)

        loss_q = torch.stack(loss_all).sum(0)
        loss_q.backward()
        optimizer.step()

        print(
            'Epoch {epoch} | Loss {loss}'.format(
                epoch=epoch,
                loss=cumulative_loss / float(len(train_loader))
            )
        )
        torch.cuda.empty_cache()

        return cumulative_loss / len(train_loader)

    def eval_loop(self, data_loader):
        """
        Evaluates the model on detection tasks sampled by data_loader
        Args:
            data_loader (torch.utils.data.DataLoader): episodic detection data loader.  It returns a tuple of size 4
            respectively containing the paths, the images, the targets and the labels

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: respectively
            precision, recall, average precision, F1 score and per-class average precision of the model
        """
        self.eval()

        batch_statistics = []
        all_labels = []

        for batch_index, (paths, images, targets, labels) in enumerate(data_loader):
            targets = self.rename_labels(targets)
            support_set, support_set_targets, query_set, query_set_targets = self.split_support_and_query_set(
                images,
                targets
            )

            _, outputs_on_query = self.set_forward(support_set, support_set_targets, query_set)
            outputs_on_query = outputs_on_query.cpu()
            outputs_on_query = non_max_suppression(
                outputs_on_query,
                conf_thres=self.objectness_threshold,
                nms_thres=self.nms_threshold
            )

            query_set_targets[:, 2:] = xywh2xyxy(query_set_targets[:, 2:]) * self.image_size
            query_set_targets = query_set_targets.cpu()

            batch_statistics += get_batch_statistics(
                outputs_on_query,
                query_set_targets,
                iou_threshold=self.iou_threshold
            )
            all_labels += labels.tolist()

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*batch_statistics))]
        precision, recall, average_precision, f1, ap_class = ap_per_class(
            true_positives,
            pred_scores,
            pred_labels,
            all_labels
        )

        return precision, recall, average_precision, f1, ap_class

    def rename_labels(self, targets):
        """

        Args:
            targets (torch.Tensor): targets given by the data loader

        Returns:
            torch.Tensor: same targets but the labels all lie in range(n_way)
        """
        old_labels = np.unique(targets[:, 1])
        labels_mapping = {}
        for new_label, old_label in enumerate(old_labels):
            labels_mapping[old_label] = new_label
        for box in targets:
            box[1] = labels_mapping[float(box[1])]

        return targets

    def split_support_and_query_set(self, images, targets):
        """
        Split images and targets between support set and query set
        Args:
            images (torch.Tensor): shape (n_way*(n_support+n_query), dim_of_img)
            targets (torch.Tensor): shape (L, 6) where L is the sum of the number of boxes in every images

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: both images and targets, split between
            support set and query set
        """
        # Split images between support set and query set
        support_set_list = []
        query_set_list = []
        support_indices = []
        query_indices = []
        for index, image in enumerate(images):
            if index % (self.n_support + self.n_query) < self.n_support:
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
            old_indices = np.unique(x_targets[:, 0])
            indices_mapping = {}
            for new_index, old_index in enumerate(old_indices):
                indices_mapping[old_index] = new_index
            for box in x_targets:
                box[0] = indices_mapping[float(box[0])]

        return (
            support_set.to(self.device),
            support_targets.to(self.device),
            query_set.to(self.device),
            query_targets.to(self.device),
        )

    def get_complete_loss_dict(self, support_loss_dict, query_loss_dict):
        """
        Merge the dictionaries containing the losses on the support set and on the query set
        Args:
            support_loss_dict (dict): contains the losses of the model on the support set
            query_loss_dict (dict): contains the losses of the model on the query set

        Returns:
            dict: merged dictionary with modified keys that say whether the loss was from the support or query set
        """
        complete_loss_dict = {}

        for key, value in support_loss_dict.items():
            complete_loss_dict['support_' + str(key)] = value

        for key, value in query_loss_dict.items():
            complete_loss_dict['query_' + str(key)] = value

        return complete_loss_dict

    def include_episode_loss_dict(self, loss_dict, episode_loss_dict, number_of_dicts):
        """
        Include the items of episode_loss_dict in loss_dict by creating a new item when the key is not in loss_dict, and
        by additioning the weighted values when the key is already in loss_dict
        Args:
            loss_dict (dict): losses of precedent layers
            episode_loss_dict (dict): losses of current layer
            number_of_dicts (int): how many dicts will be added to loss_dict in one epoch

        Returns:
            dict: updated losses
        """
        for key, value in episode_loss_dict.items():
            if key not in loss_dict:
                loss_dict[key] = episode_loss_dict[key] / float(number_of_dicts)
            else:
                loss_dict[key] += episode_loss_dict[key] / float(number_of_dicts)

        return loss_dict
