#!/usr/bin/python3
# coding=utf-8
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
global low
low=0
def Rebuild(pict):
    re=rebuild()
    re.image=cv2.cvtColor(pict,cv2.COLOR_BGR2GRAY)
    re.thresh()
    re.seperate()
    return re.seperateG

class rebuild:
    def __init__(self):
        self.image = []
        self.handle = []
        self.seperateR = []
        self.seperateG = []
        self.seperateB = []
        self.add = []
    def Getpic(self):
        filename = 'hunan.jpg'
        img = cv2.imread(filename)
        self.image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        w,h = self.image.shape
        #w=int(w)
        #h=int(h)

    def thresh(self):
        ret,thresh1=cv2.threshold(self.image,90,255,cv2.THRESH_BINARY_INV)
        kernel1 = np.ones((2,2),np.uint8)
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1)
        closing= cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)
        x,y,w,h = cv2.boundingRect(closing)
        bounding = cv2.rectangle(closing,(x,y),(x+w,y+h),(0,255,0),2)

        cropImg = bounding[y:y+h, x:x+w]
        res=[]

        i=0
        for fi in range(w):
            if not sum(cropImg[:,fi])==0:
                i=0
            if sum(cropImg[:,fi])==0 and i>5:
                res.append(fi)
            else:
                i=i+1

        #for fi in range(w):
        #   if sum(cropImg[:,fi])==0:
        #       res.append(fi)
        cropImg=np.delete(cropImg,res,axis=1)
        res=cv2.resize(cropImg,(50,50),interpolation=cv2.INTER_LINEAR)
        for fi in range(50):
            for fj in range(50):
                if not res[fi,fj]==0:
                    res[fi,fj]=255
        self.image=res


    def run(self,location,direc):
        step = 0
        x,y = location
        w,h = self.image.shape
        #w=int(w)
        #h=int(h)
        while not self.image[x,y]==low:
            x = x+direc[0]
            y = y+direc[1]
            if x<0 or x>w-1 or y<0 or y>h-1:
                break
            step = step+1
        return step
    def seperate(self):
        w,h = self.image.shape
        self.seperateR=np.zeros([w,h])
        self.seperateG=np.zeros([w,h])
        self.seperateB=np.zeros([w,h])
        self.handle=np.zeros([w,h])
        direc=[[-1,1],[0,1],[1,1],[1,0]]
        mat=[[fi,fj] for fi in range(w) for fj in range(h)]
        for location in mat:
            x,y=location
            direction=[0,0]
            if self.image[x,y]==low:
                self.handle[x,y]=direction[1]
                continue
            for fi in range(len(direc)):
                temp=self.run(location,direc[fi])#
                a,b=direc[fi]
                if temp>math.sqrt(a*a+b*b)*direction[0]:
                    direction=[temp,fi+1]
            self.handle[x,y]=direction[1]
        for location in mat:
            x,y=location
            if self.handle[x,y]==1:
                self.seperateR[x,y]=255
            if self.handle[x,y]==2 or self.handle[x,y]==4:
                self.seperateG[x,y]=255
                #print('ok')
            if self.handle[x,y]==3:
                self.seperateB[x,y]=255
    def addRGB(self):
        self.add=Image.merge("RGB",(self.seperateR,self.seperateG,self.seperateB))
    def show(self):
        plt.imshow(self.seperateB,'gray')
        plt.show()

    def start(self):
        self.Getpic()
        self.thresh()
        self.seperate()
        #self.addRGB()
        self.show()

dir="data/test1"
maindir="data_set5"
names = ["beijing","tianjin","shanghai","chongqing","hebei","shanxi","liaoning","jilin","heilongjiang"\
    ,"jiangsu","zhejiang","anhui","fujian","jiangxi","shandong","henan","hunan","hubei","guangdong",\
    "guangxi","hainan","sichuan","guizhou","yunnan","shaanxi","gansu","qinghai","taiwan","neimenggu",\
    "xizang","ningxia","xinjiang","xianggang","aomen"]
os.mkdir(maindir)
for c in range(40):
    count=c
    province=names[count]
    print('making '+province)
    save_dir=maindir+'/'+province
    le=len(province)
    os.mkdir(save_dir)
    i=0
    for root,dirs,filename in os.walk(dir):
        for fi in filename:
            if str(fi[0:le])==province:
                i=i+1
                shutil.copy(os.path.join(root,fi),save_dir)
                pre=save_dir+'/'+province+'.jpg'
                save=save_dir+'/'+province+str(i)+'.jpg'
                os.rename(pre,save)
                old = cv2.imread(save)
                new = Rebuild(old)
                cv2.imwrite(save,new)