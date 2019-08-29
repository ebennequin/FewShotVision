import numpy as np
import pytest

from classification.src import MethodEvaluation


class TestFeaturesProcessing:

    @staticmethod
    @pytest.mark.parametrize('features, labels', [
        (
            np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ]),
            np.array([0, 1, 1]),
        ),
        (
                np.array([
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [0, 0, 0, 0],
                ]),
                np.array([0, 1, 1, 3]),
        ),
        (
                np.array([
                    [4, 5, 6, 7],
                    [0, 1, 2, 3],
                    [8, 9, 10, 11],
                ]),
                np.array([1, 0, 1]),
        ),
    ])
    def test_process_features_returns_correct_class_list_when_features_and_labels_are_provided(features, labels):
        features_per_label = MethodEvaluation('CUB')._process_features(features_and_labels=(features, labels))

        assert features_per_label.keys() == {0, 1}
        assert len(features_per_label[0]) == 1
        assert len(features_per_label[1]) == 2
        np.testing.assert_array_equal(features_per_label[0][0], np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(features_per_label[1][0], np.array([4, 5, 6, 7]))
        np.testing.assert_array_equal(features_per_label[1][1], np.array([8, 9, 10, 11]))

    @staticmethod
    def test_process_features_does_not_delete_lines_with_zeros_when_not_at_the_end():
        features = np.array([
                    [0, 0, 0, 0],
                    [4, 5, 6, 7],
                ])
        labels = np.array([1, 1])
        features_per_label = MethodEvaluation('CUB')._process_features(features_and_labels=(features, labels))

        assert 1 in features_per_label.keys()
        assert len(features_per_label[1]) == 2

    @staticmethod
    def test_process_features_does_not_delete_lines_with_zero_sum_if_not_all_zeros():
        features = np.array([
                    [1, 2, 3, 4],
                    [-1, 1, 0, 0],
                ])
        labels = np.array([1, 1])
        features_per_label = MethodEvaluation('CUB')._process_features(features_and_labels=(features, labels))

        assert 1 in features_per_label.keys()
        assert len(features_per_label[1]) == 2
