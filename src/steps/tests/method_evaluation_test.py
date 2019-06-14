from src.steps import MethodEvaluation
import numpy as np

class TestMethodEvaluation:

    class TestRandomSwap:


        def random_swap_return_swap_classification(self):
            n_test_way = 5
            n_shot = 5
            n_query = 15
            n_swaps = 1
            dim_img = 10
            method_evaluation_step = MethodEvaluation(dataset="dataset", n_swaps=n_swaps, n_query=n_query, n_shot=n_shot,
                                                      n_test_way=n_test_way)
            classification_task = np.ones((n_test_way, n_shot + n_query, dim_img))
            swap_classification_task = method_evaluation_step._random_swap()

            assert True
