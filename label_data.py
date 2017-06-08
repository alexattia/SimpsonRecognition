import os 
import sys
import numpy as np
from selenium import webdriver
import glob
import urllib
import time
import keras
import cv2
import ffmpy
from random import shuffle
import matplotlib.pyplot as plt

characters = glob.glob('./characters/*')
map_characters = {0: 'abraham_grampa_simpson', 1: 'bart_simpson', 
                  2: 'charles_montgomery_burns', 3: 'homer_simpson', 4: 'krusty_the_clown',
                  5: 'lisa_simpson', 6: 'marge_simpson', 7: 'moe_szyslak', 
                  8: 'ned_flanders', 9: 'sideshow_bob'}

def get_character_name(name):
    chars = [k.split('/')[2] for k in glob.glob('./characters/*')]
    char_name = [k for k in chars if name.lower().replace(' ', '_') in k]
    if len(char_name) > 0:
        return char_name[0]
    else:
        print('FAKE NAME')
        return 'ERROR'

def labelized_data(interactive=False):
    for fname in glob.glob('./*.avi'):
        try:
            m,s = np.random.randint(0,3), np.random.randint(0,59)
            cap = cv2.VideoCapture(fname) #video_name is the video being called
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(1, fps*(m*60+s)) # Where frame_no is the frame you want
            i = 0
            while True:
                i+=1
                ret, frame = cap.read() # Read the frame
                if i % np.random.randint(150, 350) == 0:
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
                            title = './characters/%s/pic_%s_%d.jpg' % (name_char, fname.split('/')[1].split('.')[0], np.random.randint(10000))
                            cv2.imwrite(title, frame[:,:int(frame.shape[1]/2)])
                            print('Saved at %s' % title)
                            print('%s : %d photos labeled' % (name_char, len([k for k in glob.glob('./characters/%s/*' % name_char) 
                                                                            if 'pic_video' in k or 'edited' in k ])))
                    elif where.lower() in ['right', 'r']:
                        plt.close()
                        plt.imshow(frame[:,int(frame.shape[1]/2):])
                        plt.show()
                        name = input('Name ?[Name or No] ')
                        plt.close()
                        if name.lower() not in ['no','n','']:
                            name_char = get_character_name(name)
                            title = './characters/%s/pic_%s_%d.jpg' % (name_char, 
                                                           fname.split('/')[1].split('.')[0], np.random.randint(10000))
                            cv2.imwrite(title, frame[:,int(frame.shape[1]/2):])
                            print('Saved at %s' % title)
                            print('%s : %d photos labeled' % (name_char, len([k for k in glob.glob('./characters/%s/*' % name_char) 
                                                                            if 'pic_video' in k or 'edited' in k ])))
                    elif where.lower() in ['full', 'f']:
                        name = input('Name ?[Name or No] ')
                        plt.close()
                        if name.lower() not in ['no','n','']:
                            name_char = get_character_name(name)
                            title = './characters/%s/pic_%s_%d.jpg' % (name_char, 
                                                           fname.split('/')[1].split('.')[0], np.random.randint(10000))
                            cv2.imwrite(title,  frame)
                            print('Saved at %s' % title)
                            print('%s : %d photos labeled' % (name_char, len([k for k in glob.glob('./characters/%s/*' % name_char) 
                                                                            if 'pic_video' in k or 'edited' in k ])))
        except Exception as e:
            if e == KeyboardInterrupt:
                return
            else:
                continue

def generate_pic_from_videos():
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

def classify_pics(model_path):
    l = glob.glob('./autogenerate/*.jpg')
    model = keras.models.load_model(model_path)
    shuffle(l)
    d = len(l)
    for i, p in enumerate(l): 
        img = cv2.imread(p)
        img = cv2.resize(img, (64, 64)).astype('float32') / 255.
        a = model.predict(img.reshape((-1, 64, 64, 3)), verbose=0)[0]
        if np.max(a) > 0.6:
            char = map_characters[np.argmax(a)]
            os.rename(p, './autogenerate/%s/%s' % (char, p.split('/')[2]))
        else:
            os.remove(p)
        print('\r%d/%d'%(i+1, d), end='')

if __name__ == '__main__':
    labelized_data(interactive=True)

