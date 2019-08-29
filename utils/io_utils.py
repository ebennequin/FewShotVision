import glob
import os

import numpy as np
import torch

from utils import configs, backbones

model_dict = dict(
    Conv4=backbones.Conv4,
    Conv4S=backbones.Conv4S,
    Conv6=backbones.Conv6,
    ResNet10=backbones.ResNet10,
    ResNet18=backbones.ResNet18,
    ResNet34=backbones.ResNet34,
    ResNet50=backbones.ResNet50,
    ResNet101=backbones.ResNet101,
)


def path_to_step_output(dataset, backbone, method, output_dir=configs.save_dir):
    """
    Defines the path where the outputs will be saved on the disk
    Args:
        dataset (str): name of the dataset
        backbone (str): name of the backbone of the model
        method (str): name of the used method
        output_dir (str): may be common to other experiments

    Returns:
        str: path to the output of the step
    """
    checkpoint_dir = os.path.join(
        output_dir,
        dataset,
        '_'.join([method, backbone]),
    )

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def set_and_print_random_seed(random_seed, save=False, checkpoint_dir='./'):
    """
    Set and print numpy random seed, for reproducibility of the training,
    and set torch seed based on numpy random seed
    Args:
        random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
        save (bool): if True, the numpy random seed is saved in seeds.txt
        checkpoint_dir (str): output folder where the seed is saved
    Returns:
        int: numpy random seed

    """
    if random_seed is None:
        random_seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)
    torch.manual_seed(np.random.randint(0, 2**32-1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prompt = 'Random seed : {}\n'.format(random_seed)
    print(prompt)

    if save:
        with open(os.path.join(checkpoint_dir, 'seeds.txt'), 'a') as f:
            f.write(prompt)

    return random_seed


def get_path_to_json(dataset, split):
    """

    Args:
        dataset (str):  which dataset to load
        split (str): whether to use base, val or novel dataset

    Returns:
        str: path to JSON file
    """
    if dataset == 'cross':
        if split == 'base':
            path_to_json_file = configs.data_dir['miniImageNet'] + 'all.json'
        else:
            path_to_json_file = configs.data_dir['CUB'] + split + '.json'
    elif dataset == 'cross_char':
        if split == 'base':
            path_to_json_file = configs.data_dir['omniglot'] + 'noLatin.json'
        else:
            path_to_json_file = configs.data_dir['emnist'] + split + '.json'
    else:
        path_to_json_file = configs.data_dir[dataset] + split + '.json'

    return path_to_json_file


def get_assigned_file(checkpoint_dir, num):
    # TODO: returns path to .tar file corresponding to epoch num in checkpoint_dir (even if it doesn't exist)
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    # TODO: returns path to .tar file corresponding to maximal epoch in checkpoint_dir, None if checkpoint_dir is empty
    # TODO  What happens if checkpoint_dir only contains best_model.tar ?
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    # TODO returns best_model.tar in checkpoint_dir if there is one, else returns get_resume_file(checkpoint_dir)
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
