from src.steps.method_evaluation import MethodEvaluation

def test_main_does_not_return_error():
    args = [
        '--dataset',
        'omniglot',
        '--model',
        'Conv4',
        '--method',
        'baseline',
        '--n_iter',
        '1',
    ]

    MethodEvaluation().apply(args)
