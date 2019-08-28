import click
from detection.src.steps import YOLOMAMLTraining

@click.command()
@click.option('--dataset_config', default='pipelines/detection/experiments_config/coco.data')
@click.option('--model_config', default='pipelines/detection/experiments_config/deep-tiny-yolo-3-way.cfg')
@click.option('--pretrained_weights', default='./data/weights/tiny.weights')
@click.option('--n_way', default=3)
@click.option('--n_shot', default=5)
@click.option('--n_query', default=10)
@click.option('--optimizer', default='Adam')
@click.option('--learning_rate', default=0.001)
@click.option('--approx', default=True)
@click.option('--task_update_num', default=2)
@click.option('--print_freq', default=100)
@click.option('--validation_freq', default=5000)
@click.option('--n_epoch', default=2)
@click.option('--n_episode', default=4)
@click.option('--objectness_threshold', default=0.5)
@click.option('--nms_threshold', default=0.3)
@click.option('--iou_threshold', default=0.5)
@click.option('--image_size', default=208)
@click.option('--random_seed', default=None)
@click.option('--output_dir', default='./output')

def main(
        dataset_config,
        model_config,
        pretrained_weights,
        n_way,
        n_shot,
        n_query,
        optimizer,
        learning_rate,
        approx,
        task_update_num,
        print_freq,
        validation_freq,
        n_epoch,
        n_episode,
        objectness_threshold,
        nms_threshold,
        iou_threshold,
        image_size,
        random_seed,
        output_dir,
):
    """
    Initializes the YOLOMAMLTraining step and executes it.
    Args:
        dataset_config (str): path to data config file
        model_config (str): path to model definition file
        pretrained_weights (str): path to a file containing pretrained weights for the model
        n_way (int): number of labels in a detection task
        n_shot (int): number of support data in each class in an episode
        n_query (int): number of query data in each class in an episode
        optimizer (str): must be a valid class of torch.optim (Adam, SGD, ...)
        learning_rate (float): learning rate fed to the optimizer
        approx (bool): whether to use an approximation of the meta-backpropagation
        task_update_num (int): number of updates inside each episode
        print_freq (int): inside an epoch, print status update every print_freq episodes
        validation_freq (int): inside an epoch, frequency with which we evaluate the model on the validation set
        n_epoch (int): number of meta-training epochs
        n_episode (int): number of episodes per epoch during meta-training
        objectness_threshold (float): at evaluation time, only keep boxes with objectness above this threshold
        nms_threshold (float): threshold for non maximum suppression, at evaluation time
        iou_threshold (float): threshold for intersection over union
        image_size (int): size of images (square)
        random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
        output_dir (str): path to experiments output directory
    """
    step = YOLOMAMLTraining(
        dataset_config,
        model_config,
        pretrained_weights,
        n_way,
        n_shot,
        n_query,
        optimizer,
        learning_rate,
        approx,
        task_update_num,
        print_freq,
        validation_freq,
        n_epoch,
        n_episode,
        objectness_threshold,
        nms_threshold,
        iou_threshold,
        image_size,
        random_seed,
        output_dir,
    )
    step.apply()

if __name__ == '__main__':
    main()
