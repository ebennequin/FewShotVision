import os

from src.steps.fetch_model import FetchModel
from src.utils.io_utils import path_to_step_output

class TestFetchModel:

    @staticmethod
    def test_step_returns_epoch_and_state_dictionnary():
        input_path = os.path.join(path_to_step_output('CUB', 'Conv4', 'baseline', 'fake_outputs'), '0.tar')
        output = FetchModel(input_path).apply()

        assert output.keys() == {'epoch', 'state'}

    @staticmethod
    def test_step_admits_plus_character():
        input_path = os.path.join(path_to_step_output('CUB', 'Conv4', 'baseline++', 'fake_outputs'), '0.tar')
        output = FetchModel(input_path).apply()

        assert output.keys() == {'epoch', 'state'}

