import numpy as np
import pytest

from src.steps.method_evaluation import MethodEvaluation

class TestSetClassificationTask:

    @staticmethod
    @pytest.mark.parametrize(
        'n_way,n_shot,n_query,n_swaps, dim_features', [
            (10, 5, 1, 1, 10),
            (5, 1, 3, 1, 2),
            (2, 5, 10, 1, 30),
            (10, 5, 1, 2, 3),
        ]
    )
    def test_set_classification_task_output_shape(n_way, n_shot, n_query, n_swaps, dim_features):
        number_img_per_label = 20
        method_evaluation_step = MethodEvaluation(dataset="dataset", n_swaps=n_swaps, n_query=n_query,
                                                  n_shot=n_shot, test_n_way=n_way)

        labels = ["label_{k}".format(k=k) for k in range(50)]
        features_per_label = {label: np.random.rand(number_img_per_label, dim_features) for label in labels}
        classification_task = method_evaluation_step._set_classification_task(features_per_label)

        assert classification_task.shape == (n_way, n_shot + n_query, dim_features)
