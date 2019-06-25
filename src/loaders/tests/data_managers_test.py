from src.loaders.data_managers import DetectionSetDataManager

class TestDetectionSetDataManager:

    def test_returns_data_loader(self):
        data_manager = DetectionSetDataManager(3,1,2,8)
        data_loader = data_manager.get_data_loader('./data/coco/small_image_list.txt')
        item = next(iter(data_loader))
        print(data_loader)
