#coding:utf-8
import os
from PIL import Image
import numpy as np
import cv2

def resize(imgPath,savePath):
    files = os.listdir(imgPath)
    files.sort()
    print('****************')
    print('input :',imgPath)
    print('start...')
    for file in files:
        fileType = os.path.splitext(file)
        if fileType[1] == '.png':
            new_png = cv2.imread(imgPath+'/'+file) #打开图片
            # new_png = cv2.resize(new_png,(28,28))   #x改变图片大小
            gray = cv2.cvtColor(new_png, cv2.COLOR_BGR2GRAY)
            dst = 255 - gray
            #new_png = new_png.resize((20, 20),Image.ANTIALIAS) #改变图片大小
            # matrix = 255-np.asarray(new_png) #图像转矩阵 并反色
            new_png = Image.fromarray(dst) #矩阵转图像
            new_png.save(savePath+'/'+file) #保存图片
    print('down!')
    print('****************')

if __name__ == '__main__':
    # 待处理图片地址
    dataPath = 'D:/imgae classification/9.20/digit-classifier-master/NEWDATA_930'
    #保存图片的地址
    savePath = 'D:/imgae classification/9.20/digit-classifier-master/NEWDATA_930_CR'
    resize(dataPath,savePath)