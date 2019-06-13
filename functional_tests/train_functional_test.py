import os

import pytest

from src.steps.method_training import MethodTraining

class TestTrainingMethods:

    @staticmethod
    @pytest.mark.parametrize(
        'method',
        [
            'baseline',
            'baseline++',
            'protonet',
            'matchingnet',
            'relationnet',
            'relationnet_softmax',
            'maml',
            'maml_approx',
        ])
    def test_step_does_not_return_error(method):
        current_dir = os.path.dirname(__file__)
        args = dict(
            dataset='CUB',
            backbone='Conv4',
            num_classes=4412,
            stop_epoch=1,
            shallow=True,
            method=method,
            train_aug=True,
            optimizer='SGD',
            learning_rate=0.001,
            n_episode=2,
            output_dir=os.path.join(current_dir, 'tests_data')
        )

        MethodTraining(**args).apply()

    @staticmethod
    @pytest.mark.parametrize(
        'method',
        [
            'baseline',
            'protonet',
        ])
    def test_training_are_reproducible_when_using_same_seed(method):
        current_dir = os.path.dirname(__file__)
        args = dict(
            dataset='CUB',
            backbone='Conv4',
            num_classes=4412,
            stop_epoch=1,
            shallow=True,
            method=method,
            train_aug=True,
            optimizer='SGD',
            learning_rate=0.001,
            n_episode=2,
            output_dir=os.path.join(current_dir, 'tests_data'),
            random_seed=1,
        )
        training_step_1 = MethodTraining(**args)
        model_1 = training_step_1.apply()
        training_step_2 = MethodTraining(**args)
        model_2 = training_step_2.apply()

        assert model_1['epoch'] == model_2['epoch']
        assert model_1['state'].keys() == model_2['state'].keys()
        for key in model_1['state'].keys():
            assert model_1['state'][key].equal(model_2['state'][key])
