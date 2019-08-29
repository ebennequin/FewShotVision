import click
from classification.src.steps import *
from utils import configs


@click.command()
@click.option('--dataset', default='CUB')
@click.option('--backbone', default='Conv4')
@click.option('--method', default='protonet')
@click.option('--train_aug', default=True)
@click.option('--train_n_way', default=5)
@click.option('--test_n_way', default=5)
@click.option('--n_shot', default=5)
@click.option('--n_query', default=16)
@click.option('--split', default='novel')
@click.option('--save_iter', default=-1)
@click.option('--num_classes', default=4412)
@click.option('--stop_epoch', default=1)
@click.option('--start_epoch', default=0)
@click.option('--shallow', default=False)
@click.option('--resume', default=False)
@click.option('--warmup', default=False)
@click.option('--optimizer', default='Adam')
@click.option('--learning_rate', default=0.001)
@click.option('--n_episode', default=100)
@click.option('--n_iter', default=600)
@click.option('--adaptation', default=False)
@click.option('--random_seed', default=None)
@click.option('--output_dir', default=configs.save_dir)
@click.option('--n_swaps', default=0)
@click.option('--trained_model', default=None, help='If not None, fetches the trained model and skips training step')

def main(
        dataset,
        backbone,
        method,
        train_aug,
        train_n_way,
        test_n_way,
        n_shot,
        n_query,
        split,
        save_iter,
        num_classes,
        stop_epoch,
        start_epoch,
        shallow,
        resume,
        warmup,
        optimizer,
        learning_rate,
        n_episode,
        n_iter,
        adaptation,
        random_seed,
        output_dir,
        n_swaps,
        trained_model,
):
    """
    Initializes and executes the steps to train and/or evaluate a model.
    Args:
        dataset (str): CUB/miniImageNet/cross/omniglot/cross_char
        backbone (str): Conv{4|6} / ResNet{10|18|34|50|101}
        method (str): baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}
        train_n_way (int): number of labels in a classification task during training
        test_n_way (int): number of labels in a classification task during testing
        n_shot (int): number of labeled data in each class
        train_aug (bool): perform data augmentation or not during training
        shallow (bool): reduces the dataset to 256 images (typically for quick code testing)
        num_classes (int): total number of classes in softmax, only used in baseline #TODO delete this parameter
        start_epoch (int): starting epoch
        stop_epoch (int): stopping epoch
        resume (bool): continue from previous trained model with largest epoch
        warmup (bool): continue from baseline, neglected if resume is true
        optimizer (str): must be a valid class of torch.optim (Adam, SGD, ...)
        learning_rate (float): learning rate fed to the optimizer
        n_episode (int): number of episodes per epoch during meta-training
        random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
        output_dir (str): path to experiments output directory
        n_swaps (int): number of swaps between labels in the support set of each episode, in order to
        test the robustness to label noise
        split (str): which dataset is considered (base, val or novel)
        save_iter (int): save feature from the model trained in x epoch, use the best model if x is -1
        n_query (int): number of query data for each class in a classification task
        n_iter (int): number of classification tasks on which the model is tested
        adaptation (boolean): further adaptation in test time or not
        trained_model (str): path to the file containing the model parameters, must end in .tar. If not None, fetches
        the trained model and skips training step
    """
    if trained_model is not None:
        step_to_trained_model = MethodTraining(
            dataset,
            backbone,
            method,
            train_n_way,
            test_n_way,
            n_shot,
            train_aug,
            shallow,
            num_classes,
            start_epoch,
            stop_epoch,
            resume,
            warmup,
            optimizer,
            learning_rate,
            n_episode,
            random_seed,
            output_dir,
            n_swaps,
        )
    else:
        step_to_trained_model = FetchModel(trained_model)

    embedding_step = Embedding(
        dataset,
        backbone,
        method,
        train_n_way,
        test_n_way,
        n_shot,
        train_aug,
        shallow,
        split,
        save_iter,
        output_dir,
        random_seed,
    )

    evaluation_step = MethodEvaluation(
        dataset,
        backbone,
        method,
        train_n_way,
        test_n_way,
        n_shot,
        n_query,
        train_aug,
        split,
        save_iter,
        n_iter,
        adaptation,
        random_seed,
        n_swaps,
    )

    model_state = step_to_trained_model.apply()
    features_and_labels = embedding_step.apply(model_state)
    evaluation_step.apply(model_state, features_and_labels)

if __name__=='__main__':
    main()
