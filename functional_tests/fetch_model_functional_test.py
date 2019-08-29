import os

from classification.src.steps import FetchModel
from utils.io_utils import path_to_step_output

class TestFetchModel:
    current_dir = os.path.dirname(__file__)

    def test_step_returns_epoch_and_state_dictionnary(self):

        input_path = os.path.join(self.current_dir, path_to_step_output('CUB', 'Conv4', 'baseline', 'tests_data'),
                                  '0.tar')
        output = FetchModel(input_path).apply()

        assert output.keys() == {'epoch', 'state'}

    def test_step_admits_plus_character(self):
        input_path = os.path.join(self.current_dir, path_to_step_output('CUB', 'Conv4', 'baseline++', 'tests_data'),
                                  '0.tar')
        output = FetchModel(input_path).apply()

        assert output.keys() == {'epoch', 'state'}

