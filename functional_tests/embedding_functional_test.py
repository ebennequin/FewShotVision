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
            'relationnet',
            'relationnet_softmax',
        ])
    def test_step_does_not_return_error(method):
        current_dir = os.path.dirname(__file__)
        dataset = 'CUB'
        backbone = 'Conv4'

        path = os.path.join(current_dir, path_to_step_output(dataset, backbone, method, 'tests_data'), '0.tar')

        args = dict(
            dataset=dataset,
            backbone=backbone,
            method=method,
            train_aug=True,
            shallow=True,
            output_dir=os.path.join(current_dir, 'tests_data'),
        )

        model = FetchModel(path).apply()

        features, labels = Embedding(**args).apply(model)

