import click
from detection.src.steps import YOLOMAMLDetect

@click.command()
@click.option('--episode_config', help='Run create_episode.py to obtain the configuration file of an episode')
@click.option('--model_config',
              help='If the episode contains N different classes, the detector must perform N-way classification')
@click.option('--trained_weights',
              help='The weights must come from a network trained with the same .cfg file as model_config')
@click.option('--learning_rate', default=0.01, help='Should be the same as during training')
@click.option('--task_update_num', default=3, help='Should be the same as during training')
@click.option('--objectness_threshold', default=0.8)
@click.option('--nms_threshold', default=0.4)
@click.option('--iou_threshold', default=0.2)
@click.option('--image_size', default=208)
@click.option('--random_seed', default=None)
@click.option('--output_dir', default='./output/detections')

def main(
        episode_config,
        model_config,
        trained_weights,
        learning_rate,
        task_update_num,
        objectness_threshold,
        nms_threshold,
        iou_threshold,
        image_size,
        random_seed,
        output_dir,
):
    """
    Initializes the YOLOMAMLDetect step and executes it.
    Args:
        episode_config (str): path to the .data configuration file of the episode
        model_config (str): path to the .cfg file defining the structure of the YOLO model
        trained_weights (str): path to the file containing the trained weights of the model
        learning_rate (str): learning rate for weight updates on the support set
        task_update_num (str): number of weight updates on the support set
        objectness_threshold (float): the algorithm only keep boxes with a higher objectness confidence
        nms_threshold (float): non maximum suppression threshold
        iou_threshold (float): intersection over union threshold
        image_size (int): size of input images
        random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
        output_dir (str): directory where the predicted boxes are saved
    """
    step = YOLOMAMLDetect(
        episode_config,
        model_config,
        trained_weights,
        learning_rate,
        task_update_num,
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
