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
):
    training_step = MethodTraining(
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

    model_state = training_step.apply()
    features_and_labels = embedding_step.apply(model_state)
    evaluation_step.apply(model_state, features_and_labels)

if __name__=='__main__':
    main()
