import os

import h5py
from pipeline.steps import AbstractStep
import torch
import torch.optim
from torch.autograd import Variable

from src import backbone
from src.loaders.datamgr import SimpleDataManager
from src.utils import configs
from src.utils.io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file


class Embedding(AbstractStep):
    def apply(self, args):

        params = parse_args('save_features', args)

        model, data_loader, outfile = self._get_data_loader_model_and_outfile(params)

        self._save_features(model, data_loader, outfile)

    def dump_output(self, _, output_folder, output_name, **__):
        pass

    def _save_features(self, model, data_loader, outfile):
        f = h5py.File(outfile, 'w')
        max_count = len(data_loader) * data_loader.batch_size
        print(data_loader.batch_size, max_count)
        all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
        all_feats = None
        count = 0
        # TODO: here, last batch is smaller than batch_size, thus the last columns of all_feats are empty (and deleted in feature_loader.py)
        for i, (x, y) in enumerate(data_loader):
            if i % 100 == 0:
                print('{:d}/{:d}'.format(i, len(data_loader)))

            x = x.cuda()
            x_var = Variable(x)
            feats = model(x_var)
            # if i==528:
            #     print(x.size())
            #     print(feats.size())
            if all_feats is None:
                all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
            all_feats[count:count + feats.size(
                0)] = feats.data.cpu().numpy()  # TODO: why .cpu().numpy() ? probably to fit expected input of h5py dataset
            all_labels[count:count + feats.size(0)] = y.cpu().numpy()
            count = count + feats.size(0)

        count_var = f.create_dataset('count', (1,), dtype='i')
        count_var[0] = count
        f.close()

    def _get_data_loader_model_and_outfile(self, params):
        ''' Function that returns data loaders and backbone model and path to outfile

        Args:
            params: parameters returned by parse_args function from io_utils.py

        Returns:
            tuple : data_loader, modem and outfile

        '''
        # TODO: unify with train.py
        assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'

        # Defines image size
        if 'Conv' in params.model:
            if params.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84
        else:
            image_size = 224

        if params.dataset in ['omniglot', 'cross_char']:
            assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
            params.model = 'Conv4S'

        # Defines path to data
        split = params.split
        if params.dataset == 'cross':
            if split == 'base':
                loadfile = configs.data_dir['miniImagenet'] + 'all.json'
            else:
                loadfile = configs.data_dir['CUB'] + split + '.json'
        elif params.dataset == 'cross_char':
            if split == 'base':
                loadfile = configs.data_dir['omniglot'] + 'noLatin.json'
            else:
                loadfile = configs.data_dir['emnist'] + split + '.json'
        else:
            loadfile = configs.data_dir[params.dataset] + split + '.json'

        # Compute checkpoint directory and fetch model parameters
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'
        if not params.method in ['baseline', 'baseline++']:
            checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

        if params.save_iter != -1:
            modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
        elif params.method in ['baseline', 'baseline++']:
            modelfile = get_resume_file(checkpoint_dir)
        else:
            modelfile = get_best_file(checkpoint_dir)

        # Defines output file for computed features
        if params.save_iter != -1:
            outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                                   split + "_" + str(params.save_iter) + ".hdf5")
        else:
            outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split + ".hdf5")

            # Return data loader TODO: why do we do batches here ?
        datamgr = SimpleDataManager(image_size, batch_size=64)
        data_loader = datamgr.get_data_loader(loadfile, aug=False, shallow=params.shallow)

        # Create backbone
        if params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                model = backbone.Conv4NP()
            elif params.model == 'Conv6':
                model = backbone.Conv6NP()
            elif params.model == 'Conv4S':
                model = backbone.Conv4SNP()
            else:
                model = model_dict[params.model](flatten=False)
        elif params.method in ['maml', 'maml_approx']:
            raise ValueError('MAML do not support save feature')
        else:
            model = model_dict[params.model]()

        # Load trained parameters into backbone and delete all non-feature layers
        model = model.cuda()
        tmp = torch.load(modelfile)
        state = tmp['state']
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.",
                                     "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        model.load_state_dict(state)
        model.eval()

        dirname = os.path.dirname(outfile)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        return (model,
                data_loader,
                outfile,
                )
