# ML experimentations with Document Content

## About

This repository contains a set of experiments with document content. In particular it tries to identify content such as tables.
It also contains attempts to pre-process documents in new ways.

## Content

### Table detection

For table segmentation UNet is used. The model is trained on a dataset of 10000 syntetically generated images containing text and tables.
Generator code can be found in [main.py](main.py). The model is trained from scratch to see overall performance.
Overall performance is satisfactory when inference is done on other synthetic images.
Real documents representation is below expectations, as a next step for improvement, real documents should be added into the dataset.
The model was trained using Google Collab GPU (40GB) for 10 epochs, it took 6 minutes to train and achieve validation cross-entropy loss of 0.0098.
GPU utilization was 90-95% during training with batch size of 128 and 8 parallel workers for the data loaders.

#### Demo

In [demo](demo) folder you can find a Gradio app that allows you to upload an image and see the results of the model.

### Document representation

The idea is to represent a document as an uneven plane, where each point represnts gravitation pull of elements in the document.
It is a very rough representation, but it can be used to find the most important parts of the document.
