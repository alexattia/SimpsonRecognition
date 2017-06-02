# SimpsonRecognition

Training a Convolutional Neural Network to recognize The Simpson TV Show characters using Keras (TensorFlow backend).  
See the blog article on [Medium](https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras-d8e1796eae36).  

### First part : Collecting data

The first part is collecting and labeling Simpson pictures.  
Most of the pictures are from Simpson video, analyzed frame by frame.

Run ``python3 label_data.py`` into a folder with Simpson episodes (.avi format) to analyze them and label frames.  
You crop each frame (left part, right part, full-frame, nothing) and then label it.  

### Second part : Training with Keras

The second part is training the model. We keep only characters with more than 300 pictures (it's a minimum, I aim to have 1000 pictures per category/character). My goal is to have 20 classes.  
As you can see on the Jupyter notebook, I benchmark two models : 4 and 6 convolutional layers neural networks. Because of the small number of pictures (approx. 1k pictures per class), I use data augmentation.  
Currently, I have 95% of accuracy for 10 classes.  


![Lisa picture](https://github.com/alexattia/SimpsonRecognition/blob/master/pics/mapple_lisa.png)