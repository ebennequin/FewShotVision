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
