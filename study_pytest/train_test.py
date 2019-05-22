from train import (
    get_data_loaders_model_and_train_parameterers,
    main,
    parse_args,
)


def test_get_data_loaders_model_and_train_parameterers_return_default_value_for_epochs():
    args = [
        '--dataset',
        'omniglot',
        '--model',
        'Conv4',
        '--method',
        'baseline',
        '--num_classes',
        '4412',
    ]

    params = parse_args('train', args)

    base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params = (
        get_data_loaders_model_and_train_parameterers(params)
    )

    assert start_epoch == 0
    assert stop_epoch == 5


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
        '--shallow',
        'True',
    ]

    main(args)
