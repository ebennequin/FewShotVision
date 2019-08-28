import numpy as np

from detection.src.loaders.data_manager import DetectionSetDataManager, DetectionTaskSampler
from detection.src.yolov3.utils.datasets import ListDataset

class TestDetectionSetDataManager:

    def test_returns_data_loader(self):
        n_way = 2
        n_support = 1
        n_query = 1
        n_episode = 4

        data_manager = DetectionSetDataManager(n_way, n_support, n_query, n_episode, 416)
        data_loader = data_manager.get_data_loader(
            'detection/src/loaders/tests/small_image_list.txt',
            'detection/src/loaders/tests/images_per_label.pkl'
        )
        item = next(iter(data_loader))

        assert len(item[0]) == n_way*(n_support+n_query)
        assert item[3].shape[0] == n_way

        labels = np.unique(item[2][:, 1].numpy())
        assert len(labels) == n_way == item[3].shape[0]


class TestDataset:


    class TestDetectionTaskSampler:

        def test_get_images_per_label_returns_dict(self):
            sampler = DetectionTaskSampler(ListDataset('./src/loaders/tests/small_image_list.txt'), 2, 2, 1, 3)
            dico = sampler.images_per_label
            assert len(dico) == 15

        def test_labels_in_label_list_are_sufficiently_represented(self):
            sampler = DetectionTaskSampler(ListDataset('./src/loaders/tests/small_image_list.txt'), 2, 2, 1, 3)
            label_list = sampler.label_list
            images_per_label = sampler.images_per_label
            for label in label_list:
                assert len(images_per_label[label]) >= 3
