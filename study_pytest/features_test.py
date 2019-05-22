from scripts.save_features import (
    main,
)

def test_main_does_not_return_error():
    args = [
        '--dataset',
        'omniglot',
        '--model',
        'Conv4',
        '--method',
        'baseline',
    ]

    main(args)