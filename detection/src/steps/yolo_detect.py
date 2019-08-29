import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np
import torch
from PIL import Image



from detection.src.yolov3.model import Darknet
from detection.src.yolov3.utils.datasets import ListDataset
from detection.src.yolov3.utils.parse_config import parse_data_config
from detection.src.yolov3.utils.utils import non_max_suppression, rescale_boxes, load_classes


class YOLODetect():
    """
    This step performs detection on a given episode using a trained YOLO model.
    """

    def __init__(
            self,
            episode_config,
            model_config,
            trained_weights,
            objectness_threshold=0.8,
            nms_threshold=0.4,
            iou_threshold=0.2,
            image_size=416,
            output_dir='output/detections',
    ):
        """

        Args:
            episode_config (str): path to the .data configuration file of the episode
            model_config (str): path to the .cfg file defining the structure of the YOLO model
            trained_weights (str): path to the file containing the trained weights of the model
            objectness_threshold (float): the algorithm only keep boxes with a higher objectness confidence
            nms_threshold (float): non maximum suppression threshold
            iou_threshold (float): intersection over union threshold
            image_size (int): size of input images
            output_dir (str): directory where the predicted boxes are saved
        """
        self.data_config = parse_data_config(episode_config)
        self.model_config = model_config
        self.trained_weights = trained_weights
        self.objectness_threshold = objectness_threshold
        self.nms_threshold = nms_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.output_dir = output_dir

        self.labels = self.parse_labels(self.data_config['labels'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def apply(self):
        """
        Executes YOLODetect step and saves the result images
        """
        model = self.get_model()
        paths, images = self.get_episode()

        _, outputs = model.forward(images)

        outputs = outputs.cpu()
        outputs = non_max_suppression(
            outputs,
            conf_thres=self.objectness_threshold,
            nms_thres=self.nms_threshold
        )

        self.save_detections(list(paths), outputs)

    def dump_output(self, *_, **__):
        pass

    def parse_labels(self, labels_str):
        """
        Gets labels from a string
        Args:
            labels_str (str): string from the data config file describing the labels of the episode

        Returns:
            list: labels of the episode
        """
        labels_str_split = labels_str.split(', ')
        labels = [int(label) for label in labels_str_split]

        return labels


    def get_episode(self):
        """
        TODO: change from episodic collate_fn to standard collate_fn
        Returns:
            Tuple[Tuple, torch.Tensor, torch.Tensor]: the paths, images and target boxes of data instances composing
            the episode described in the data configuration file
        """
        dataset = ListDataset(
            list_path=self.data_config['eval'],
            img_size=self.image_size,
            augment=False,
            multiscale=False,
            normalized_labels=True,
        )

        data_instances = [dataset[i] for i in range(len(dataset))]

        paths, images, _ = dataset.collate_fn(data_instances)

        return paths, images.to(self.device)

    def get_model(self):
        """
        Returns:
            Darknet: model
        """

        return Darknet(self.model_config, self.image_size, self.trained_weights).to(self.device)


    def save_detections(self, paths, output):
        """
        Draws predicted boxes on input images and saves them in self.output_dir
        Args:
            paths (list): paths to input images
            output (list): output of the model. Each element is a torch.Tensor of shape (number_of_kept_detections, 7).
            Each detection contains (x1, y1, x2, y2, objectness_confidence, class_score, class_predicted)
        """
        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print('Saving images:')
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(paths, output)):

            print('Image {index}: {path}'.format(index=img_i, path=path))

            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, self.image_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                classes = load_classes(self.data_config['names'])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print('\t+ Label: {label_name}, Classif conf: {class_conf}'.format(
                        label_name=classes[int(cls_pred)],
                        class_conf=cls_conf.item())
                    )

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color='white',
                        verticalalignment='top',
                        bbox={'color': color, 'pad': 0},
                    )

            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = path.split('/')[-1].split('.')[0] + '.png'
            plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', pad_inches=0.0)
            plt.close()

