import pytest


from src.steps.embedding import Embedding


@pytest.fixture
def args_without_method():
    return [
        '--dataset',
        'CUB',
        '--model',
        'Conv4',
        # '--shallow',
        # 'True',
        '--method',
    ]


def test_baseline(args_without_method):
    Embedding().apply(args_without_method + ['baseline'])


def test_baseline_p(args_without_method):
    Embedding().apply(args_without_method + ['baseline++'])


def test_protonet(args_without_method):
    Embedding().apply(args_without_method + ['protonet'])


def test_matchingnet(args_without_method):
    Embedding().apply(args_without_method + ['matchingnet'])


def test_relationnet(args_without_method):
    Embedding().apply(args_without_method + ['relationnet'])


def test_relationnet_softmax(args_without_method):
    Embedding().apply(args_without_method + ['relationnet_softmax'])
