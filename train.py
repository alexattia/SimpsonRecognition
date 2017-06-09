import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import h5py
from sklearn.model_selection import train_test_split
import glob
from random import shuffle
from collections import Counter
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam

# map_characters = {0: 'abraham_grampa_simpson', 1: 'bart_simpson', 
#                   2: 'charles_montgomery_burns', 3: 'homer_simpson', 4: 'krusty_the_clown',
#                   5: 'lisa_simpson', 6: 'marge_simpson', 7: 'moe_szyslak', 
#                   8: 'ned_flanders', 9: 'sideshow_bob'}

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
    3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'homer_simpson', 
    6: 'krusty_the_clown', 7: 'lisa_simpson', 8: 'marge_simpson', 
    9: 'milhouse_van_houten', 10: 'moe_szyslak', 11: 'ned_flanders', 
    12: 'principal_skinner', 13: 'sideshow_bob'}

pic_size = 64
batch_size = 32
epochs = 200
num_classes = len(map_characters)
pictures_per_class = 1000
test_size = 0.15

def load_pictures(BGR):
    pics = []
    labels = []
    for k, char in map_characters.items():
        pictures = [k for k in glob.glob('./characters/%s/*' % char) if 'edited' in k 
                                                                     or 'pic_vid' in k]
        nb_pic = round(pictures_per_class/(1-test_size)) if round(pictures_per_class/(1-test_size))<len(pictures) else len(pictures)
        for pic in np.random.choice(pictures, nb_pic):
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (pic_size,pic_size))
            pics.append(a)
            labels.append(k)
    return np.array(pics), np.array(labels) 

def get_dataset(save=False, load=False, BGR=False):
    if load:
        h5f = h5py.File('dataset.h5','r')
        X = h5f['dataset'][:]
        h5f.close()    

        h5f = h5py.File('labels.h5','r')
        y = h5f['labels'][:]
        h5f.close()    
    else:
        X, y = load_pictures(BGR)
        y = keras.utils.to_categorical(y, num_classes)
        if save:
            h5f = h5py.File('dataset.h5', 'w')
            h5f.create_dataset('dataset', data=X)
            h5f.close()

            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('labels', data=y)
            h5f.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
    if not load:
        dist = {k:tuple(d[k] for d in [dict(Counter(np.where(y_train==1)[1])), dict(Counter(np.where(y_test==1)[1]))]) 
                for k in range(num_classes)}
        print('\n'.join(["%s : %d train pictures & %d test pictures" % (map_characters[k], v[0], v[1]) 
            for k,v in sorted(dist.items(), key=lambda x:x[1][0], reverse=True)]))
    return X_train, X_test, y_train, y_test

def create_model_four_conv(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
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

def create_model_six_conv(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, opt

def lr_schedule(epoch):
    lr = 0.01
    return lr*(0.1**int(epoch/10))

def training(model, X_train, X_test, y_train, y_test, data_augmentation=True, callback=False, six_conv=False):
    if data_augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        if six_conv:
            history = model.fit_generator(datagen.flow(X_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            epochs=40,
                            validation_data=(X_test, y_test),
                            callbacks=[LearningRateScheduler(lr_schedule),
                                        ModelCheckpoint('model_6conv.h5',save_best_only=True)])
        else:
            if callback:
                filepath="weights.best.hdf5"
                checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
                callbacks_list = [checkpoint]
                history = model.fit_generator(datagen.flow(X_train, y_train,
                                             batch_size=batch_size),
                                            steps_per_epoch=X_train.shape[0] // batch_size,
                                            epochs=epochs,
                                            callbacks=callbacks_list,
                                            validation_data=(X_test, y_test))
            else: 
                history = model.fit_generator(datagen.flow(X_train, y_train,
                             batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(X_test, y_test))
    else:
        history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)
    return model, history
