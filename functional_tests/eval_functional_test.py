import os

import pytest

from classification.src import load_features_and_labels_from_file
from classification.src import MethodEvaluation
from classification.src import FetchModel
from utils.io_utils import path_to_step_output

class TestEvaluation:

    current_dir = os.path.dirname(__file__)

    @pytest.mark.parametrize(
        'method',
        [
            'baseline',
            'baseline++',
            'protonet',
            'matchingnet',
            'relationnet',
            'relationnet_softmax',
        ])
    def test_step_does_not_return_error(self, method):
        dataset = 'CUB'
        backbone = 'Conv4'

        args = dict(
            dataset=dataset,
            backbone=backbone,
            method=method,
            train_aug=True,
            n_iter=2,
            n_swaps=2,
        )

        path_to_model = os.path.join(self.current_dir, path_to_step_output(dataset, backbone, method, 'tests_data'), '0.tar')
        model = FetchModel(path_to_model).apply()

        path_to_features = os.path.join(self.current_dir, path_to_step_output(dataset, backbone, method, 'tests_data'),
                                        'novel.hdf5')
        features, labels = load_features_and_labels_from_file(path_to_features)

        results = MethodEvaluation(**args).apply(model, (features, labels))

    @pytest.mark.parametrize(
        'method',
        [
            'maml',
            'maml_approx',
        ])
    def test_step_does_not_return_error_for_maml(self, method):
        dataset = 'CUB'
        backbone = 'Conv4'

        args = dict(
            dataset=dataset,
            backbone=backbone,
            method=method,
            train_aug=True,
            n_iter=2
        )

        path_to_model = os.path.join(self.current_dir, path_to_step_output(dataset, backbone, method, 'tests_data'),
                                     'best_model.tar')
        model = FetchModel(path_to_model).apply()

        results = MethodEvaluation(**args).apply(model, None)

    @pytest.mark.parametrize(
        'method',
        [
            'protonet',
            'matchingnet',
            'relationnet',
        ])
    def test_step_does_not_change_input_model(self, method):
        dataset = 'CUB'
        backbone = 'Conv4'

        args = dict(
            dataset=dataset,
            backbone=backbone,
            method=method,
            train_aug=True,
            n_iter=2
        )

        path_to_model = os.path.join(self.current_dir, path_to_step_output(dataset, backbone, method, 'tests_data'),
                                     'best_model.tar')
        model_1 = FetchModel(path_to_model).apply()

        path_to_features = os.path.join(self.current_dir,
                                        path_to_step_output(dataset, backbone, method, 'tests_data'),
                                        'novel.hdf5')
        features, labels = load_features_and_labels_from_file(path_to_features)

        MethodEvaluation(**args).apply(model_1, (features, labels))

        model_2 = FetchModel(path_to_model).apply()

        assert model_1['epoch'] == model_2['epoch']
        assert model_1['state'].keys() == model_2['state'].keys()
        for key in model_1['state'].keys():
            assert model_1['state'][key].equal(model_2['state'][key])

    @pytest.mark.parametrize(
        'method',
        [
            'protonet',
            'matchingnet',
            'relationnet',
        ])
    def test_returns_same_acc_mean_on_same_model_when_random_seed_is_the_same(self, method):
        dataset = 'CUB'
        backbone = 'Conv4'

        args = dict(
            dataset=dataset,
            backbone=backbone,
            method=method,
            train_aug=True,
            n_iter=2,
            random_seed=1,
        )

        path_to_model = os.path.join(self.current_dir, path_to_step_output(dataset, backbone, method, 'tests_data'),
                                     'best_model.tar')
        model_state = FetchModel(path_to_model).apply()

        path_to_features = os.path.join(self.current_dir, path_to_step_output(dataset, backbone, method, 'tests_data'),
                                        'novel.hdf5')
        features, labels = load_features_and_labels_from_file(path_to_features)

        results_1 = MethodEvaluation(**args).apply(model_state, (features, labels))
        results_2 = MethodEvaluation(**args).apply(model_state, (features, labels))

        assert results_1 == results_2
