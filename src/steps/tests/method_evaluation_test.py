import pytest

from src.steps import MethodEvaluation
import numpy as np

class TestMethodEvaluation:

    class TestRandomSwap:

        @staticmethod
        @pytest.mark.parametrize(
            'test_n_way,n_shot,n_query,n_swaps, dim_features', [
                (10, 5, 1, 1, 3),
                (5, 1, 3, 1, 1),
                (2, 5, 10, 1, 3),
                (10, 5, 1, 2, 3),
            ]
        )
        def test_random_swap_return_swap_classification(test_n_way, n_shot, n_query, n_swaps, dim_features):
            test_n_way = 3
            n_shot = 3
            n_query = 1
            n_swaps = 1
            dim_features = 3
            method_evaluation_step = MethodEvaluation(dataset="dataset", n_swaps=n_swaps, n_query=n_query,
                                                      n_shot=n_shot, test_n_way=test_n_way)
            classification_task = np.random.rand(test_n_way, n_shot + n_query, dim_features)
            swap_classification_task = method_evaluation_step._random_swap(classification_task)
            diff_classification_task = swap_classification_task - classification_task
            assert swap_classification_task.shape == (test_n_way, n_shot+n_query, dim_features)
            if n_swaps == 1:
                assert np.count_nonzero(diff_classification_task) == 2*dim_features
            assert np.count_nonzero(diff_classification_task[:, n_shot:]) == 0
