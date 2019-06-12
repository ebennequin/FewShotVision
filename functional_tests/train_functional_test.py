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
            stop_epoch= 1,
            shallow=True,
            method=method,
            train_aug=True,
            optimizer='Adam',
            learning_rate=0.001,
            n_episode=2,
            output_dir=os.path.join(current_dir, 'tests_data')
        )

        MethodTraining(**args).apply()
