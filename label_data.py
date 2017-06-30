import os 
import sys
import numpy as np
from selenium import webdriver
import glob
import urllib
import train
import time
import keras
import cv2
import ffmpy
from random import shuffle
import matplotlib.pyplot as plt

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak', 
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}
pic_size = 64


def get_character_name(name):
    """
    Get the character name from just a part of it, comparing to saved characters
    :param name: part of the character name
    :return: full name
    """
    chars = [k.split('/')[2] for k in glob.glob('./characters/*')]
    char_name = [k for k in chars if name.lower().replace(' ', '_') in k]
    if len(char_name) > 0:
        return char_name[0]
    else:
        print('FAKE NAME')
        return 'ERROR'

def labelized_data(to_shuffle=False, interactive=False):
    """
    Interactive labeling data with the possibility to crop the picture shown : full picture,
    left part, right part. Manually labeling data from .avi videos in the same folder. Analzying
    frame (randomly chosen) of each video and then save the picture into the right character 
    folder.
    :param interactive: boolean to label from terminal
    """
    movies = glob.glob('./*.avi')
    if to_shuffle:
        shuffle(movies)
    for fname in movies[::-1]:
        try:
            m,s = np.random.randint(0,3), np.random.randint(0,59)
            cap = cv2.VideoCapture(fname) #video_name is the video being called
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(1, fps*(m*60+s)) # Where frame_no is the frame you want
            i = 0
            while True:
                i+=1
                ret, frame = cap.read() # Read the frame
                # Resizing HD pictures (we don't need HD)
                if np.min(frame.shape[:2]) > 900:
                    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
                if i % np.random.randint(100, 250) == 0:
                    if interactive:
                        f = plt.ion()
                    plt.imshow(frame)
                    plt.show()
                    where = input('Where is the character ?[No,Right,Left,Full] ')
                    if where.lower() == 'stop':
                        # os.remove(fname)
                        raise
                    
                    elif where.lower() in ['left', 'l']:
                        plt.close()
                        plt.imshow(frame[:,:int(frame.shape[1]/2)])
                        plt.show()
                        name = input('Name ?[Name or No] ')
                        plt.close()
                        if name.lower() not in ['no','n','']:
                            name_char = get_character_name(name)
                            name_new_pic = 'pic_{:04d}.jpg'.format(len(glob.glob('./characters/%s/*' % name_char)))
                            title = './characters/%s/%s' % (name_char, name_new_pic)
                            cv2.imwrite(title, frame[:,:int(frame.shape[1]/2)])
                            print('Saved at %s' % title)
                            print('%s : %d photos labeled' % (name_char, len(glob.glob('./characters/%s/*' % name_char))))
                    
                    elif where.lower() in ['right', 'r']:
                        plt.close()
                        plt.imshow(frame[:,int(frame.shape[1]/2):])
                        plt.show()
                        name = input('Name ?[Name or No] ')
                        plt.close()
                        if name.lower() not in ['no','n','']:
                            name_char = get_character_name(name)
                            name_new_pic = 'pic_{:04d}.jpg'.format(len(glob.glob('./characters/%s/*'% name_char)))
                            title = './characters/%s/%s' % (name_char, name_new_pic)
                            cv2.imwrite(title, frame[:,int(frame.shape[1]/2):])
                            print('Saved at %s' % title)
                            print('%s : %d photos labeled' % (name_char, len(glob.glob('./characters/%s/*' % name_char))))
                    
                    elif where.lower() in ['full', 'f']:
                        name = input('Name ?[Name or No] ')
                        plt.close()
                        if name.lower() not in ['no','n','']:
                            name_char = get_character_name(name)
                            name_new_pic = 'pic_{:04d}.jpg'.format(len(glob.glob('./characters/%s/*'% name_char)))
                            title = './characters/%s/%s' % (name_char, name_new_pic)
                            cv2.imwrite(title,  frame)
                            print('Saved at %s' % title)
                            print('%s : %d photos labeled' % (name_char, len(glob.glob('./characters/%s/*'% name_char))))
        except Exception as e:
            if e == KeyboardInterrupt:
                return
            else:
                continue

def generate_pic_from_videos():
    """
    Randomly generate pictures from videos : get the full picture, the right part, the left part.
    So, three pictures are saved for each analyzed frame (chosen randomly).
    """
    for k, fname in enumerate(glob.glob('./*.avi')):
        m,s = np.random.randint(0,3), np.random.randint(0,59)
        cap = cv2.VideoCapture(fname) 
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(1, fps*(m*60+s)) # Where frame_no is the frame you want    
        i = 0
        while i < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            try:
                i+=1
                ret, frame = cap.read() # Read the frame
                if i % np.random.randint(400, 700) == 0:
                    pics = {'pic_%s_r_%d_%d.jpg' % (fname.split('/')[1].split('.')[0], 
                            i, np.random.randint(10000)):frame[:,:int(frame.shape[1]/2)],
                            'pic_%s_l_%d_%d.jpg' % (fname.split('/')[1].split('.')[0], 
                            i, np.random.randint(10000)): frame[:,int(frame.shape[1]/2):],
                            'pic_%s_f_%d_%d.jpg' % (fname.split('/')[1].split('.')[0], 
                            i, np.random.randint(10000)): frame}
                    for name, img in pics.items():
                        cv2.imwrite('./autogenerate/' + name, img)
            except:
                pass
        print('\r%d/%d' % (k+1, len(glob.glob('./*.avi'))), end='')

def classify_pics():
    """
    Use a Keras saved model to classify pictures and move them into the right character folder.
    """
    l = glob.glob('./autogenerate/*.jpg')
    model = train.load_model_from_checkpoint('./models/weights.best_6conv2.hdf5', six_conv=True)
    d = len(l)
    for i, p in enumerate(l): 
        img = cv2.imread(p)
        img = cv2.resize(img, (pic_size, pic_size)).astype('float32') / 255.
        a = model.predict(img.reshape((-1, pic_size, pic_size, 3)), verbose=0)[0]
        if np.max(a) > 0.6:
            char = map_characters[np.argmax(a)]
            os.rename(p, './autogenerate/%s/%s' % (char, p.split('/')[2]))
        else:
            os.remove(p)
        print('\r%d/%d'%(i+1, d), end='')

if __name__ == '__main__':
    labelized_data(interactive=True)

