from src.steps.method_evaluation import MethodEvaluation

def test_main_does_not_return_error():

    args_without_method = [
            '--dataset',
            'omniglot',
            '--model',
            'Conv4',
            '--n_iter',
            '1',
            '--method',
    ]

    methods=[
        'baseline',
        'baseline++',
        'protonet',
        'matchingnet',
        'relationnet',
        'relationnet_softmax',
        'maml',
        'maml_approx',
    ]

    for method in methods:
        MethodEvaluation().apply(args_without_method+[method])
