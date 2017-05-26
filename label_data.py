import os 
import sys
import numpy as np
from selenium import webdriver
import glob
import urllib
import time
import cv2
import ffmpy
import matplotlib.pyplot as plt

characters = glob.glob('./characters/*')

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
                        os.remove(fname)
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

if __name__ == '__main__':
    labelized_data(interactive=True)

