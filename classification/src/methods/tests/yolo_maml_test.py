import torch

from detection.src.yolo_maml import YOLOMAML

class TestYOLOMAML:

    class TestSplitSupportAndQuerySet:

        def test_split_is_done_correctly(self):
            n_way = 2
            n_support = 1
            n_query = 2

            model = YOLOMAML('dummy', n_way, n_support, n_query, 416)

            images = torch.randn((n_way*(n_support+n_query), 3, 2, 2))
            targets = torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [2, 0, 0, 0, 0, 0],
                    [3, 1, 0, 0, 0, 0],
                    [3, 0, 0, 0, 0, 0],
                    [4, 1, 0, 0, 0, 0],
                    [4, 1, 0, 0, 0, 0],
                    [5, 1, 0, 0, 0, 0],
                ], dtype=torch.float
            )

            support_set, support_targets, query_set, query_targets = model.split_support_and_query_set(images, targets)

            assert support_set.shape[0] == n_way * n_support
            assert query_set.shape[0] == n_way * n_query

            true_support_targets = torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                ], dtype=torch.float
            )

            true_query_targets = torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [2, 1, 0, 0, 0, 0],
                    [2, 1, 0, 0, 0, 0],
                    [3, 1, 0, 0, 0, 0],
                ], dtype=torch.float
            )

            assert torch.all(torch.eq(support_targets, true_support_targets))
            assert torch.all(torch.eq(query_targets, true_query_targets))

    class TestRenameLabels:

        def test_labels_are_correct(self):
            n_way = 2
            n_support = 1
            n_query = 2

            model = YOLOMAML('dummy', n_way, n_support, n_query, 416)

            targets = torch.tensor(
                [
                    [0, 3, 0, 0, 0, 0],
                    [0, 18, 0, 0, 0, 0],
                    [1, 3, 0, 0, 0, 0],
                    [2, 3, 0, 0, 0, 0],
                    [3, 18, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 0],
                    [4, 24, 0, 0, 0, 0],
                    [4, 24, 0, 0, 0, 0],
                    [5, 18, 0, 0, 0, 0],
                ], dtype=torch.float
            )

            true_targets = torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [2, 0, 0, 0, 0, 0],
                    [3, 1, 0, 0, 0, 0],
                    [3, 0, 0, 0, 0, 0],
                    [4, 2, 0, 0, 0, 0],
                    [4, 2, 0, 0, 0, 0],
                    [5, 1, 0, 0, 0, 0],
                ], dtype=torch.float
            )

            new_targets = model.rename_labels(targets)

            assert torch.all(torch.eq(new_targets, true_targets))
