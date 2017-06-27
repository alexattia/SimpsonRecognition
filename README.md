# SimpsonRecognition

Training a Convolutional Neural Network to recognize The Simpson TV Show characters using Keras (TensorFlow backend).  
To have more detailed explanations, see the blog article on [Medium](https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras-d8e1796eae36). If you like it, don't hesitate to recommend it (as for the dataset on [Kaggle](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset))  

### First part : Collecting data

The first part is collecting and labeling Simpson pictures.  
Most of the pictures are from Simpson video, analyzed frame by frame.

Run ``python3 label_data.py`` into a folder with Simpson episodes (.avi format) to analyze them and label frames.  
You crop each frame (left part, right part, full-frame, nothing) and then label it.  

You can find the dataset on [Kaggle](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)

### Second part : Training with Keras

The second part is training the model. My goal is to have 20 classes. I aim to have 1000 pictures per class, unfortunately some characters are not often on screen so I have fewer pictures for those characters.
As you can see on the Jupyter notebook, I benchmark two models : 4 and 6 convolutional layers neural networks. Because of the small number of pictures (approx. 1k pictures per class), I use data augmentation.  
Currently, I have 96% of accuracy (F1-Score) for 18 classes.  

![Lisa picture](https://github.com/alexattia/SimpsonRecognition/blob/master/pics/mapple_lisa.png)

### Third part : Faster R-CNN

The third part is upgrade the model to detect and recognize characters. I have to annotate data to get bounding boxes for characters for each picture in order to train a new model : [Faster R-CNN](https://arxiv.org/abs/1506.01497)(which is based on a Region Proposal Network).  
