import click
from detection.src.steps import YOLOMAMLCreateEpisode

@click.command()
@click.option('--dataset_config', default='detection/configs/coco.data')
@click.option('--n_way', default=3)
@click.option('--n_shot', default=5)
@click.option('--n_query', default=10)
@click.option('--output_dir', default='./data/coco/episodes')
@click.option('--episode_name', default=None)
@click.option('--labels', default=None)

def main(
        dataset_config,
        n_way,
        n_shot,
        n_query,
        output_dir,
        episode_name,
        labels,
):
    """
    Initializes and executes the YOLOMAMLCreateEpisode step
    Args:
        dataset_config (str): path to data config file
        n_way (int): number of different classes in the episode
        n_shot (int): number of support set instances per class
        n_query (int): number of query set instances per class
        output_dir (str): output directory
        episode_name (str): name of the output file (without type extension)
        labels (list): labels from which to sample instances. If None, labels will be sampled at random.
    """
    step = YOLOMAMLCreateEpisode(
        dataset_config,
        n_way,
        n_shot,
        n_query,
        output_dir,
        episode_name,
        labels,
    )
    step.apply()

if __name__ == '__main__':
    main()
