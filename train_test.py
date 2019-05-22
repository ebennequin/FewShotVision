from train import main


def test_main_does_not_return_error():
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
    ]

    main(args)
