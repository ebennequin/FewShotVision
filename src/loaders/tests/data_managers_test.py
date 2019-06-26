import numpy as np

from src.loaders.data_managers import DetectionSetDataManager

class TestDetectionSetDataManager:

    def test_returns_data_loader(self):
        n_way = 2
        n_support = 1
        n_query = 1
        n_episode = 4

        data_manager = DetectionSetDataManager(n_way, n_support, n_query, n_episode)
        data_loader = data_manager.get_data_loader(
            './src/loaders/tests/small_image_list.txt',
            './src/loaders/tests/images_per_label.pkl'
        )
        item = next(iter(data_loader))

        assert len(item[0]) == n_way*(n_support+n_query)

        labels = np.unique(item[2][:, 1].numpy())
        # TODO assert len(labels) == n_way
