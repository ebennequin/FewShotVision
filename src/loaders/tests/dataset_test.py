from src.loaders.dataset import EpisodicBatchSampler, DetectionTaskSampler
from src.utils.io_utils import set_and_print_random_seed
from src.yolov3.utils.datasets import ListDataset

import torch

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

    class TestDetectionTaskSampler:

        def test_get_images_per_label_returns_dict(self):
            sampler = DetectionTaskSampler(ListDataset('./data/coco/small_image_list.txt'), 5, 6, 10, 100)
            dico = sampler._get_images_per_label()
            assert len(dico) == 56
            assert dico[24][1] == 47
