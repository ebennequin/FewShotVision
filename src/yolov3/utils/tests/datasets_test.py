from src.yolov3.utils.datasets import ListDataset
from torch.utils.data import DataLoader

class TestListDataset:

    def test_list_dataset_has_correct_shape(self):
        list_path = './data/coco/5k.txt'

        dataset = ListDataset(list_path)

        item = dataset[8]
        assert type(item[0]) is str
        assert len(item[1].shape) == 3
        assert len(item[2].shape) == 2
        assert item[2].shape[1] == 6

    def test_data_loader_has_expected_shape(self):

        list_path = './data/coco/5k.txt'

        dataset = ListDataset(list_path)

        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        temp = next(iter(dataloader))

