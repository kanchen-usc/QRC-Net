# QRC-Net

This repository contains the implementation of *Query-guided Regression Network with Context Policy for Phrase Grounding* in [ICCV 2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Query-Guided_Regression_Network_ICCV_2017_paper.pdf).

## Setup

*Note*: Please read the feature representation files in ```feature``` and ```annotation``` directories before using the code.

**Platform:** Tensorflow-1.0.1<br/>
**Visual features:** We use [Faster-RCNN](https://github.com/endernewton/tf-faster-rcnn) fine-tuned on [Flickr30K Entities](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/). Afer fine-tuning, please put visual features in the ```feature``` directory (More details can be seen in the [```README.md```](./feature/README.md) in this directory).<br/>
**Sentence features:** We encode one-hot vector for each query, as well as the annotation for each query and image pair. Pleae put the encoded features in the ```annotation``` directory (More details are provided in the [```README.md```](./annotation/README.md) in this directory).<br/>
**File List:** We generate a file list for each image in the Flickr30K Entities. If you would like to train and test on other dataset (e.g. [Referit Game](http://tamaraberg.com/referitgame/)), please follow the similar format in the ```flickr_train_val.lst``` and ```flickr_test.lst```.<br/>
**Hyper parameters:** Please check the ```Config``` class in the ```train.py```.

## Training & Test

For training, please enter the root folder of ```QRC-Net```, then type
```
$ python train.py -m [Model Name] -g [GPU ID]
```
For testing, please entre the root folder of ```QRC-Net```, then type
```
$ python evaluate.py -m [Model Name] -g [GPU ID] --restore_id [Restore epoch ID]
```
Make sure the model name entered for evaluation is the same as the model name in training, and the epoch id exists.

## Citation

If you find the repository is useful for your research, please consider citing the following work:
```
@InProceedings{Chen_2017_ICCV,
author = {Chen, Kan* and Kovvuri, Rama* and Nevatia, Ram}, 
title = {Query-guided Regression Network with Context Policy for Phrase Grounding},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
year = {2017} 
}
```