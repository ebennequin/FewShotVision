import os

from pipeline.steps import AbstractStep
import pickle

from src.utils import configs

from src.loaders.dataset import create_dict_images_per_label
from src.yolov3.utils.datasets import ListDataset


class YOLOMAMLCreateDic(AbstractStep):
    """
    This step create the dictionary requires for yolo MAML
    """

    def __init__(
            self,
            file_path='data/coco/trainvalno5k.txt',
            output_dir=configs.save_dir,
    ):
        """
        Args:
            dataset (str): path to data config file
            output_dir (str): path to experiments output directory
        """

        self.file_path = file_path
        self.output_dir = output_dir

    def apply(self):
        """
        Execute the YOLOMAMLCreateDic step

        """

        dataset = ListDataset(self.file_path)
        dict_images_per_label = create_dict_images_per_label(dataset)
        file_name = os.path.join(self.output_dir, 'dict_images_per_label.pkl')
        with open(file_name, 'wb') as handle:
            pickle.dump(dict_images_per_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_output(self, _, output_folder, output_name, **__):
        pass
