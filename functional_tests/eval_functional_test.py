import os

import numpy as np
import pytest

from src.loaders.feature_loader import load_features_and_labels_from_file
from src.steps.method_evaluation import MethodEvaluation
from src.steps.fetch_model import FetchModel
from src.utils.io_utils import path_to_step_output

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
            n_iter=2
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
