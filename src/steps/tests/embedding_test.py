from src.steps.embedding import Embedding

def test_main_does_not_return_error():
    args = [
        '--dataset',
        'omniglot',
        '--model',
        'Conv4',
        '--method',
        'baseline',
    ]

    Embedding().apply(args)
