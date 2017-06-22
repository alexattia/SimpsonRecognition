import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

n_elem = 5
with open('./annotation.txt') as f:
    already_labeled = [k.strip().split(' ')[0] for k in f.readlines()]

target = open('annotation.txt', 'a')
pics = glob.glob('./characters/milhouse_van_houten/*.*')
for p in pics[:10]:
    if p not in already_labeled:
        im = cv2.imread(p)
        ax = plt.gca()
        fig = plt.gcf()
        implot = ax.imshow(im)

        position = []
        def onclick(event):
            if event.xdata != None and event.ydata != None:
                position.append((event.xdata, event.ydata)) 
                n_clicks = len(position)
                if n_clicks == 2:
                    line = '{0} {1} {2}'.format(p, 
                        ' '.join([str(int(k)) for k in position[0]]), 
                        ' '.join([str(int(k)) for k in position[1]])) 
                    target.write(line)
                    target.write("\n")
                    plt.close()   
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
plt.close()

