import os

from pipeline.steps import AbstractStep
import torch

from utils import configs

class FetchModel(AbstractStep):
    """
    This step returns the state of a trained model in a dictionary, from the path to a .tar file
    """
    def __init__(self, path_to_model):
        """

        Args:
            path_to_model (str): path to the file containing the model parameters, must end in .tar
        """
        self.path_to_model = path_to_model

    def apply(self):
        """
        Execute the steps
        Returns:
            dict: contains the model state
        """
        assert self.path_to_model.endswith(".tar"), "Input path to file must end in .tar"
        model_state = torch.load(self.path_to_model)

        torch.save(model_state, os.path.join(configs.save_dir, 'loaded_model.tar'))

        return model_state

    def dump_output(self, _, output_folder, output_name, **__):
        pass
