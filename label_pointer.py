"""
Label pictures and get bounding boxes coordinates (upper left and lower right).
Using mouse click we can get those coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from random import shuffle
import os
from train import map_characters

# List of already labeled pictures
with open('./annotation.txt') as f:
    already_labeled = [k.strip().split(' ')[0] for k in f.readlines()]

characters = map_characters.values()
shuffle(characters)

for char in characters:
    pics = glob.glob('./characters/%s/*.*' % char)
    for p in pics:
        if p not in already_labeled:
            im = cv2.imread(p)
            
            ax = plt.gca()
            fig = plt.gcf()
            implot = ax.imshow(im)
            position = []
            def onclick(event):
                """
                If click, add the mouse position to the list.
                Closing the plotted picture after 2 clicks (= 2 cornes.)
                Write the position for each picture into the text file.
                """
                if event.xdata != None and event.ydata != None:
                    position.append((event.xdata, event.ydata))
                    n_clicks = len(position)
                    if n_clicks == 2:
                        if position[0] == position[1]:
                            r = input('Delete this picture[Y/n] ? ')
                            if r.lower() in ['yes','y']:
                                os.remove(p)
                                return
                        line = '{0} {1} {2}'.format(p, 
                            ' '.join([str(int(k)) for k in position[0]]), 
                            ' '.join([str(int(k)) for k in position[1]])) 

                        # Open the annotations file to continue to write
                        target = open('annotation.txt', 'a')
                        # Write picture and coordinates
                        target.write(line)
                        target.write("\n")
                        plt.close()   
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
    plt.close()

