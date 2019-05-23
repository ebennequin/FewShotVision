from src.steps.method_training import MethodTraining


def test_step_does_not_return_error():
    args = [
        '--dataset',
        'omniglot',
        '--model',
        'Conv4',
        '--method',
        'baseline',
        '--num_classes',
        '4412',
        '--stop_epoch',
        '1',
        '--shallow',
        'True',
    ]

    MethodTraining().apply(args)
