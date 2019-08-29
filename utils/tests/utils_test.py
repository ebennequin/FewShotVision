import pytest
import torch

from utils.utils import random_swap_numpy, random_swap_tensor
import numpy as np

class TestMethodEvaluation:

    class TestRandomSwap:

        @staticmethod
        @pytest.mark.parametrize(
            'n_way,n_shot,n_query,n_swaps, dim_features', [
                (10, 5, 1, 1, 3),
                (5, 1, 3, 1, 1),
                (2, 5, 10, 1, 3),
                (10, 5, 1, 2, 3),
            ]
        )
        def test_random_swap_return_swap_classification_for_numpy(n_way, n_shot, n_query, n_swaps, dim_features):
            classification_task = np.random.rand(n_way, n_shot + n_query, dim_features)
            swap_classification_task = random_swap_numpy(classification_task, n_swaps=n_swaps, n_shot=n_shot)
            diff_classification_task = swap_classification_task - classification_task
            assert swap_classification_task.shape == (n_way, n_shot+n_query, dim_features), "Wrong shape"
            if n_swaps == 1:
                assert np.count_nonzero(diff_classification_task) == 2*dim_features
            assert np.count_nonzero(diff_classification_task[:, n_shot:]) == 0

        @staticmethod
        @pytest.mark.parametrize(
            'n_way,n_shot,n_query,n_swaps, dim_features', [
                (10, 5, 1, 1, 3),
                (5, 1, 3, 1, 1),
                (2, 5, 10, 1, 3),
                (10, 5, 1, 2, 3),
            ]
        )
        def test_random_swap_return_swap_classification_for_torch(n_way, n_shot, n_query, n_swaps, dim_features):
            classification_task = torch.randn((n_way, n_shot + n_query, dim_features, dim_features))
            swap_classification_task = random_swap_tensor(classification_task, n_swaps=n_swaps, n_shot=n_shot)
            diff_classification_task = swap_classification_task - classification_task
            assert swap_classification_task.shape == (n_way, n_shot+n_query, dim_features, dim_features)
            if n_swaps == 1:
                assert np.count_nonzero(diff_classification_task) == 2*dim_features**2
            assert np.count_nonzero(diff_classification_task[:, n_shot:]) == 0
