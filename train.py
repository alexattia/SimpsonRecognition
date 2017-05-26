import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import h5py
from sklearn.model_selection import train_test_split
import glob
from collections import Counter
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

characters = [k.split('/')[2] for k in glob.glob('./characters/*') if len([p for p in glob.glob(k+'/*') 
                                                                           if 'edited' in p or 'pic_vid' in p]) > 300]
map_characters = dict(enumerate(characters))
pic_size = 64
num_classes = len(map_characters)

def load_pictures():
    pics = []
    labels = []
    for k, char in map_characters.items():
        pictures = glob.glob('./characters/%s/*' % char)
        for pic in pictures:
            a = cv2.imread(pic)
            a = cv2.resize(a, (pic_size,pic_size))
            pics.append(a)
            labels.append(k)
    return np.array(pics), np.array(labels) 

def get_dataset(save=False):
    X, y = load_pictures()
    y = keras.utils.to_categorical(y, num_classes)
    print(X.shape, y.shape)
    if save:
        h5f = h5py.File('dataset.h5', 'w')
        h5f.create_dataset('dataset', data=X)
        h5f.close()

        h5f = h5py.File('labels.h5', 'w')
        h5f.create_dataset('labels', data=y)
        h5f.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print('Train :')
    print('\n'.join(["%s : %d pictures" % (map_characters[k], v) 
        for k,v in sorted(Counter(np.where(y_train==1)[1]).items(), key=lambda x:x[1], reverse=True)]))
    return X_train, X_test, y_train, y_test

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt
