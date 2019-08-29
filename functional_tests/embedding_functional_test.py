import os

import numpy as np
import pytest

from classification.src import Embedding
from classification.src import FetchModel
from utils.io_utils import path_to_step_output

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

    @staticmethod
    @pytest.mark.parametrize(
        'method',
        [
            'baseline',
            'protonet',
        ])
    def test_output_is_the_same_when_input_is_the_same(method):
        current_dir = os.path.dirname(__file__)
        dataset = 'CUB'
        backbone = 'Conv4'

        path = os.path.join(current_dir, path_to_step_output(dataset, backbone, method, 'tests_data'), '0.tar')

        args = dict(
            dataset=dataset,
            backbone=backbone,
            method=method,
            train_aug=True,
            shallow=False,
            output_dir=os.path.join(current_dir, 'tests_data'),
            random_seed=1,
        )

        model = FetchModel(path).apply()

        features1, labels1 = Embedding(**args).apply(model)
        features2, labels2 = Embedding(**args).apply(model)



        assert np.array_equal(features1, features2)
