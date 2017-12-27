import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

path = sys.path[0]

FILE = path+'/x2.jpg'
MARK = path+'/y2.jpg'

img = cv2.imread(FILE)
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1=res2


img = cv2.imread(MARK)
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
img2=res2
'''
img1 = cv2.imread(FILE,0)          # queryImage
img2 = cv2.imread(MARK,0) # trainImage
'''
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
# 比值测试，首先获取与 A 距离最近的点 B （最近）和 C （次近），只有当 B/C
# 小于阈值时（ 0.75 ）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为 0
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
	if m.distance < 0.9*n.distance:
		matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
singlePointColor = (110,0,0),
matchesMask = matchesMask,
flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()
