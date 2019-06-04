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
            # TODO 'maml', 'maml_approx',
        ])
    def test_step_does_not_return_error(method):
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
        )

        MethodTraining(**args).apply()
