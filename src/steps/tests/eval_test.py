import pytest

from src.steps.method_evaluation import MethodEvaluation


class TestEvaluation:

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
            dataset='omniglot',
            backbone='Conv4',
            method=method,
            train_aug=True,
            n_iter=1,
        )

        MethodEvaluation(**args).apply()
