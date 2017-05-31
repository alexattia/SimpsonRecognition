# SimpsonRecognition

Training a Convolutional Neural Network to recognize The Simpson TV Show characters using Keras (TensorFlow backend).  

### First part : Collecting data

The first part is collecting and labeling Simpson pictures.  
Most of the pictures are from Simpson video, analyzed frame by frame.

Run ``python3 label_data.py`` into a folder with Simpson episodes (.avi format) to analyze them and label frames.  
You crop each frame (left part, right part, full-frame, nothing) and then label it.  

### Second part : Training with Keras

The second part is training the model. We keep only characters with more than 300 pictures (this threshold will be higher when I will have more labeled pictures). My goal is to have 20 classes.  
Currently, the model is 4 convolutional layers neural network. Because of the small number of pictures (approx. 1k pictures per class), I use data augmentation.   


![Lisa picture](https://github.com/alexattia/SimpsonRecognition/blob/master/pics/mapple_lisa.png)