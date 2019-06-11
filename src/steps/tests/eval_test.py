import os
import pytest

from src.loaders.feature_loader import load_features_and_labels_from_file
from src.steps.method_evaluation import MethodEvaluation
from src.steps.fetch_model import FetchModel
from src.utils.io_utils import path_to_step_output

class TestEvaluation:

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
            n_iter=2
        )

        path_to_model = os.path.join(path_to_step_output(dataset, backbone, method, 'fake_outputs'), '0.tar')
        model = FetchModel(path_to_model).apply()

        path_to_features = os.path.join(path_to_step_output(dataset, backbone, method, 'fake_outputs'), 'novel.hdf5')
        features, labels = load_features_and_labels_from_file(path_to_features)

        results = MethodEvaluation(**args).apply(model, (features, labels))