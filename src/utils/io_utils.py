import numpy as np
import os
import glob
from src import backbones
from src.utils import configs

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
    checkpoint_dir = os.path.join(
        output_dir,
        dataset,
        '_'.join([method, backbone]),
    )

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


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
