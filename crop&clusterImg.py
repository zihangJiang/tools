import cv2
import numpy as np
import os
from PIL import Image
from math import *
from matplotlib import pyplot as plt
filename = 'C://Users//Administrator//Desktop//model//handwriting//test1//11.jpg'
img = cv2.imread(filename)
#----------------------旋转图片---------------------
def rotation(image_name,degree):
    height,width=image_name.shape[:2]
    if degree==0:
        return image_name
    else:
        heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
        widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))
        matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
        matRotation[0,2] +=(widthNew-width)/2
        matRotation[1,2] +=(heightNew-height)/2
        imgRotation=cv2.warpAffine(image_name,matRotation,(widthNew,heightNew),borderValue=(255,255,255))
        return imgRotation
img=rotation(img,0)
#----------------------框选省份-----------------------
coordinate = []
for fj in range(4):
    for fi in range(10):
        coordinate.append((109+219*fi, 670+278*fj, 180, 180))
#-------------------------标出框选范围----------------------
for (x, y, w, h) in coordinate:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#---------------------显示框选效果-------------------------
def show(pic):
    plt.imshow(pic,'gray')
    plt.xticks([]),plt.yticks([])
    plt.show()
show(img)
#---------------------分割保存及命名----------------------
names = ["beijing","tianjin","shanghai","chongqing","hebei","shanxi","liaoning","jilin","heilongjiang"\
    ,"jiangsu","zhejiang","anhui","fujian","jiangxi","shandong","henan","hunan","hubei","guangdong",\
    "guangxi","hainan","sichuan","guizhou","yunnan","shaanxi","gansu","qinghai","taiwan","neimenggu",\
    "xizang","ningxia","xinjiang","xianggang","aomen","hefei","hangzhou","taiyuan","shijiazhuang","wuhan"\
    ,"nanjing"]
def classify(image_name,cimg):
    if coordinate:
        #将图片剪裁保存在save_dir目录下。
        #Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
        save_dir = image_name.split('.')[0]+"prov"
        os.mkdir(save_dir)
        count = 0
        for (x1,y1,x2,y2) in coordinate:
            file_name = os.path.join(save_dir,names[count]+".jpg")
            pimg = Image.fromarray(cimg)  #opencv转化为PIL格式
            cropImg = pimg.crop((x1,y1,x1+x2,y1+y2))
            cropImg.save(file_name)
            count+=1
classify(filename,img)