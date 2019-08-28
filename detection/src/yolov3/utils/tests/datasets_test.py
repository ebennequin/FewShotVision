from detection.src.yolov3.utils.datasets import ListDataset


class TestListDataset:

    def test_list_dataset_has_correct_shape(self):
        list_path = './src/loaders/tests/small_image_list.txt'

        dataset = ListDataset(list_path)

        item = dataset[8]
        assert type(item[0]) is str
        assert len(item[1].shape) == 3
        assert len(item[2].shape) == 2
        assert item[2].shape[1] == 6
