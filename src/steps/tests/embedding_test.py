import pytest

from src.steps.embedding import Embedding


class TestEmbedding:

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
        )

        Embedding(**args).apply()
