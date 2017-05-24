import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/jeffreylutz/t/t1/IMG/center_2017_04_07_12_47_39_206.jpg'
                 )
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = img[70:(160-20), 0:320]

h,w = img.shape[:2]
# img = cv2.resize(img, (32,16), interpolation=cv2.INTER_CUBIC )

# plt.imshow(img)

# plt.show()

fifo = list()
fifo.append(1)