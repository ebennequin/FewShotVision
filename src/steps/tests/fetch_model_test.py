import pytest

from src.steps.fetch_model import FetchModel

class TestFetchModel:

    @staticmethod
    def test_step_returns_epoch_and_state_dictionnary():
        input_path='fake_outputs/CUB/Conv4/baseline_aug/0.tar'
        output = FetchModel(input_path).apply()

        assert output.keys() == {'epoch', 'state'}

    @staticmethod
    def test_step_admits_plus_character():
        input_path = 'fake_outputs/CUB/Conv4/baseline++_aug/0.tar'
        output = FetchModel(input_path).apply()

        assert output.keys() == {'epoch', 'state'}

