from classification.src import EpisodicBatchSampler
from utils.io_utils import set_and_print_random_seed


class TestDataset:

    class TestEpisodicBatchSampler:

        def test_returns_same_sampler_when_torch_seed_is_same(self):
            set_and_print_random_seed(1)
            sample_1 = list(EpisodicBatchSampler(10, 2, 3))

            set_and_print_random_seed(1)
            sample_2 = list(EpisodicBatchSampler(10, 2, 3))

            assert len(sample_1) == len(sample_2)

            for tensor_1, tensor_2 in zip(sample_1, sample_2):
                assert tensor_1.equal(tensor_2)

