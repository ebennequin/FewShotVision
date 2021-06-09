# Few Shot Learning

### ⚠️ This code is no longer maintained. For up-to-date and documented code for few-shot learning, check out [EasyFSL](https://github.com/sicara/easy-few-shot-learning)

This code was used for research in Few-Shot Image Classification and Few-Shot Object Detections. The resulting paper is available [here](https://arxiv.org/abs/1909.13579).
These two tasks constitute the two parts of the repository. It contains:

- An end to end implementation of several meta-learning algorithms for Few-Shot Image Classification, including:
 - Matching Networks
 - Prototypical Networks
 - Relation Network
 - MAML
- A first (still non working) implementation of the YOLOMAML algorithm.

It will be useful for training and testing meta-learning algorithms, allowing full reproducibility of the results. The code is extensively documented.

## Environment
Mostly:
 - Python3
 - [Pytorch](http://pytorch.org/) 1.1
 - CUDA 10

## Getting started

Git clone the repo:

```
git clone git@github.com:ebennequin/FewShotLearning.git
```

Then `cd FewShotLearning` and install and activate virtualenv:

```
virtualenv venv --python=python3
source venv/bin/activate
```

Then install dependencies. Run `pip install -r requirements.txt`.

## Download data

### Classification

#### CUB
* run `source ./download_data/scripts/download_CUB.sh`
You will need wget for that script.

#### mini-ImageNet
* run `source ./download_data/scripts/download_miniImagenet.sh`

(WARNING: This would download the 155G ImageNet dataset.) The compress file of mini-ImageNet is on muaddib.

#### Self-defined setting
* Require three data split json file: 'base.json', 'val.json', 'novel.json' for each dataset  
* The format should follow   
{"label_names": ["class0","class1",...], "image_names": ["filepath1","filepath2",...],"image_labels":[l1,l2,l3,...]}  
See test.json for reference
* Put these file in the same folder and change data_dir['DATASETNAME'] in configs.py to the folder path  

### Detection: COCO

 - run `source ./download_data/scripts/get_coco_dataset.sh`

## Few-Shot Image Classification

### Run experiment
All the parameters of the experiment (dataset, backbone, method, number of examples per class ...) can be customized.
For instance, to launch a training using Prototypical Networks on CUB, run:
```
python classification/scripts/run_experiment.py --dataset=CUB --method=protonet
```

### Outputs \& results
You can find the outputs of your experiment in `./output/{dataset}/{method}_{backbone}/`
- The `.tar` files contain the state of the trained model after different number of epochs. `best_model.tar` contains the model with the best validation accuracy, which is used for evaluation.
- The `.hdf5` file contains the features vector of all images in the evaluation dataset, along with their labels.
- `results.txt` contains the evaluation accuracy of the model.

Notes:
- Baseline and baseline++ don't save a `best_model.tar` file, and use the model with the highest number of epochs for evaluation.
- MAML-like algorithms don't save feature vectors.
 
## Few-Shot Object Detection: YOLOMAML

### Configuration
In `detection/configs`, the `.data` files configure the dataset.
The `.cfg` files configure the structure of the YOLO network.
You are free to write your own configuration.
Keep in mind that the structure of the network depends on the shape of your detection task:
if you want the few-shot detector to perform N-way classification, you must make sure that, in your config file:
 - for each `[yolo]` layer, you have `classes={N}`
 - for each `[convolutional]` layer preceeding a `[yolo]` layer, you have `filters={3*(N+5)}`

### Training
To train a YOLOMAML, run `python detection/scripts/run_yolomaml_train.py`.
You can specify many options, which are detailed in the docstring.
For instance, if you want to custom the optimizer, run `python detection/scripts/run_yolomaml_train.py --optimizer=SGD`

You can find the trained weights of the model in `output/final.weights`, and use them to perform detections.
In the `output` folder, you can also find the events file to visualize the evolution of the loss during the training
in a Tensorboard.

### Solve a few-shot object detection task
To solve such a task, you first need to sample one.
You may of course customize your task (all options are detailed in the docstring).
You can generate a 5-way 5-shot task by running `python detection/scripts/run_create_episode.py --n_way=5 --n_shot=5`
The configuration file of your episode is dumped in `data/coco/episodes` by default.

To perform detection on an episode, you need to specify
 - the configuration file of the episode
 - the configuration file of the network
 - the trained weights of the network
You need to make sure that the weights correspond to the structure of the network 
(i.e. that the training process was done with the same configuration).
Also, if your episode contains N different classes, you need to have a network fit to perform N-way classification.

If you created the episode `task-0-1-2.data` and trained a model with `deep-tiny-yolo-3-way.cfg`,
you can perform detection on this task by running:

```
python detection/scripts/run_yolomaml_detect.py 
--episode_config=data/coco/episodes/task-0-1-2.data 
--model_config=detection/configs/deep-tiny-yolo-3-way.cfg 
--trained_weights=output/final.weights
```

The detections will be dumped in `output/detections`.

## Test
- To launch the unit tests run `make test`.

- To run the functional test, run `make functional_test`. The functional tests are based on non-versioned data stored
in `data/CUB` that can be downloaded from `source ./scripts/downloaders/download_CUB.sh`. Besides, some functional
tests are based on data generated by other tests. Thus you need to run twice in a row these functional tests for them
to pass

## References
The code for Few Shot Image Classification is modified from https://github.com/wyharveychen/CloserLookFewShot.

The code for YOLOv3 is modified from https://github.com/eriklindernoren/PyTorch-YOLOv3.

The code for YOLOMAML is my own.
