from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np
import torch
from pipeline.steps import AbstractStep

from src.utils import configs

from src.methods import YOLOMAML
from src.yolov3.model import Darknet
from src.yolov3.utils.datasets import ListDataset
from src.yolov3.utils.parse_config import parse_data_config
from src.yolov3.utils.utils import non_max_suppression, xywh2xyxy, get_batch_statistics, ap_per_class, rescale_boxes, \
    load_classes


class YOLOMAMLDetect(AbstractStep):

    def __init__(
            self,
            episode_config,
            model_config,
            trained_weights,
            learning_rate,
            task_update_num,
            objectness_threshold=0.8,
            nms_threshold=0.4,
            iou_threshold=0.2,
            image_size=416,
            random_seed=None,
            output_dir=configs.save_dir,
    ):
        self.data_config = parse_data_config(episode_config)
        self.model_config = model_config
        self.trained_weights = trained_weights
        self.learning_rate = learning_rate
        self.task_update_num = task_update_num
        self.objectness_threshold = objectness_threshold
        self.nms_threshold = nms_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.random_seed = random_seed
        self.output_dir = output_dir

        self.labels = self.parse_labels(self.data_config['labels'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def apply(self):
        model = self.get_model()
        paths, images, targets = self.get_episode()

        support_set, support_targets, query_set, query_targets = model.split_support_and_query_set(images, targets)

        _, query_output = model.set_forward(support_set, support_targets, query_set)

        query_output = query_output.cpu()
        query_output = non_max_suppression(
            query_output,
            conf_thres=self.objectness_threshold,
            nms_thres=self.nms_threshold
        )

        query_targets[:, 2:] = xywh2xyxy(query_targets[:, 2:]) * self.image_size
        query_targets = query_targets.cpu()

        batch_statistics = get_batch_statistics(
            query_output,
            query_targets,
            iou_threshold=self.iou_threshold
        )

        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*batch_statistics))]
        precision, recall, average_precision, f1, ap_class = ap_per_class(
            true_positives,
            pred_scores,
            pred_labels,
            self.labels,
        )

        self.save_detections(list(paths), query_output)


    def dump_output(self, _, output_folder, output_name, **__):
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

        data_instances = [dataset[-label-1] for label in self.labels]
        data_instances.extend([dataset[i] for i in range(len(dataset))])

        paths, images, targets, _ = dataset.collate_fn(data_instances)

        return paths, images, targets

    def get_model(self):
        """
        Returns:
            YOLOMAML: meta-model
        """

        base_model = Darknet(self.model_config, self.image_size, self.trained_weights)

        model = YOLOMAML(
            base_model,
            self.data_config['n_way'],
            self.data_config['n_shot'],
            self.data_config['n_query'],
            self.image_size,
            approx=True,
            task_update_num=self.task_update_num,
            train_lr=self.learning_rate,
            objectness_threshold=self.objectness_threshold,
            nms_threshold=self.nms_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
        )

        return model

    def save_detections(self, paths, output):
        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(paths, output)):

            print("(%d) Image: '%s'" % (img_i, path))

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
                bbox_colors = np.random.choice(colors, n_cls_preds)
                classes = load_classes(self.data_config['names'])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = path.split("/")[-1].split(".")[0]
            plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
            plt.close()
