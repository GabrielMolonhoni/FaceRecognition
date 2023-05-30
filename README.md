# FaceRecognition
Face Recognition using Facenet

This work must apply Neural Networks to have face recognition.

The overview of the project can be seen at: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

Participants:
* Beatriz Andrade Luiz - CP3016307
* Gabriel Marques Molonhoni - CP3016129

## 1. Objective

Solve a face recognition problem using neural networks. Given an image of a person's face, your model must correctly identify whose face it is (out of 83 possible people).

## 2. Dataset
For model development, we considered the PubFig83 dataset. The available dataset contains the following specifications:

* Color images of the faces of 83 artists, totaling 13,840 images extracted from the Internet.
* All images were previously resized to 100x100 pixels, having been aligned according to the position of the people's eyes.

## 3. Model
We'll be using Facenet as feature extractor.

The Facenet is a TensorFlow implementation of the face recognizer described in the paper "FaceNet: A Unified Embedding for Face Recognition and Clustering". 
The project also uses ideas from the paper "Deep Face Recognition" from the Visual Geometry Group at Oxford.
The repository of Facenet can be seen at: https://github.com/davidsandberg/facenet

However, for versioning issues, we are gonna use the keras implementation (https://pypi.org/project/keras-facenet/).

On this version, differently from the original from the David Sandberg, this version returns a list of 512 features (embeddings), instead of 128.

## Obs: We used face detection (MTCNN) to crop the images and saved on Disk. The trained images were the cropped ones.
To crop the images, we used the crop_face.ipynb.
The face detection was used later on the Test Dataset, but used 'live'.
