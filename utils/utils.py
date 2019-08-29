import torch
import numpy as np


def random_swap_numpy(classification_task, n_swaps, n_shot):
    """
    Apply self.n_swaps swaps randomly to the support set
    Args:
        classification_task (ndarray): shape=(n_way, n_shot + n_query, dim_of_data) one classification
        task, which is composed of a support set and a query set
        n_swaps (int): number of swaps to execute
        n_shot (int): how many images per class are in the support set

    Returns:
        ndarray: shape=(n_way, n_shot + n_query, dim_of_data) same classification task, with swaped
        elements.
    """
    n_way = len(classification_task)
    result = classification_task.copy()
    support_set = result[:, :n_shot]
    for _ in range(n_swaps):
        swaped_classes = np.random.choice(n_way, size=2, replace=False)
        swaped_images = np.random.choice(n_shot, size=2, replace=True)
        support_set_buffer = support_set[
                                 swaped_classes[1],
                                 swaped_images[1]
                             ].copy(), support_set[
                                 swaped_classes[0],
                                 swaped_images[0]
                             ].copy()
        support_set[
            swaped_classes[0],
            swaped_images[0]
        ], support_set[
            swaped_classes[1],
            swaped_images[1]
        ] = support_set_buffer

    return result

def random_swap_tensor(classification_task, n_swaps, n_shot):
    """
    Apply self.n_swaps swaps randomly to the support set
    Args:
        classification_task (torch.Tensor): shape=(n_way, n_shot + n_query, dim_of_data) one classification
        task, which is composed of a support set and a query set
        n_swaps (int): number of swaps to execute
        n_shot (int): how many images per class are in the support set

    Returns:
        torch.Tensor: shape=(n_way, n_shot + n_query, dim_of_data) same classification task, with swaped
        elements.
    """
    n_way = len(classification_task)
    result = classification_task.clone()
    support_set = result[:, :n_shot]
    for _ in range(n_swaps):
        swaped_classes = np.random.choice(n_way, size=2, replace=False)
        swaped_images = np.random.choice(n_shot, size=2, replace=True)
        support_set_buffer = support_set[
                                 swaped_classes[1],
                                 swaped_images[1]
                             ].clone(), support_set[
                                 swaped_classes[0],
                                 swaped_images[0]
                             ].clone()
        support_set[
            swaped_classes[0],
            swaped_images[0]
        ], support_set[
            swaped_classes[1],
            swaped_images[1]
        ] = support_set_buffer

    return result


def get_complete_loss_dict(list_of_support_loss_dicts, query_loss_dict):
    """
    Merge the dictionaries containing the losses on the support set and on the query set
    Args:
        list_of_support_loss_dicts (list): element of index i contains the losses of the model on the support set
        after i weight updates
        query_loss_dict (dict): contains the losses of the model on the query set

    Returns:
        dict: merged dictionary with modified keys that say whether the loss was from the support or query set
    """
    complete_loss_dict = {}

    for task_step, support_loss_dict in enumerate(list_of_support_loss_dicts):
        for key, value in support_loss_dict.items():
            complete_loss_dict['support_' + str(key) + '_' + str(task_step)] = value

    for key, value in query_loss_dict.items():
        complete_loss_dict['query_' + str(key)] = value

    return complete_loss_dict


def include_episode_loss_dict(loss_dict, episode_loss_dict, number_of_dicts):
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
    new_loss_dict = {}
    for key, value in episode_loss_dict.items():
        if key not in loss_dict:
            new_loss_dict[key] = episode_loss_dict[key] / float(number_of_dicts)
        else:
            new_loss_dict[key] = loss_dict[key] + (episode_loss_dict[key] / float(number_of_dicts))

    return new_loss_dict


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  ) 

    return np.mean(cl_sparsity) 
