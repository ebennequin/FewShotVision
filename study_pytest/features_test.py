from save_features import (
    main,
    parse_args,
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