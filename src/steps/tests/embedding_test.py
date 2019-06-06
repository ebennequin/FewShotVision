import os
import pytest

from src.steps.embedding import Embedding
from src.steps.fetch_model import FetchModel
from src.utils.io_utils import path_to_step_output

class TestEmbedding:

    @staticmethod
    @pytest.mark.parametrize(
        'method',
        [
            'baseline',
            'baseline++',
            'protonet',
            'matchingnet',
            # TODO 'relationnet', 'relationnet_softmax',
            # TODO 'maml', 'maml_approx',
        ])
    def test_step_does_not_return_error(method):
        dataset = 'CUB'
        backbone = 'Conv4'

        args = dict(
            dataset=dataset,
            backbone=backbone,
            method=method,
            train_aug=True,
        )
        path = os.path.join(path_to_step_output(dataset, backbone, method, 'fake_outputs'), '0.tar')

        model = FetchModel(path).apply()

        features, labels = Embedding(**args).apply(model)

