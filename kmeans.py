import cv2
import numpy as np
import sys
import os
import random
import json

path = sys.path[0]
image_id="img_1.jpg"
p=path+"//"+image_id
# read in image
img=cv2.imread(p)
# reshape image
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# choose K
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
# show image
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
