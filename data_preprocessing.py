import numpy as np
import os
from PIL import Image
import tensorflow as tf
import skimage.io as io
from skimage import transform

x_train = np.empty((792,28*28))
y_label = np.empty(792)
cnt = 1

def changeImg(name):
    baseDir = 'data/'+name+'/'
    global  cnt,x_train,y_label
    for filename in os.listdir(r'data/'+name):
        if filename != '.DS_Store':
            print('Start Transfer IMG: '+baseDir + filename)
            image = Image.open(baseDir + filename)
            image = image.resize((96,96),Image.BILINEAR)
            image = image.convert("L")
            img_nparray = np.asarray(image,dtype='float32')/255
            width,hight = image.size
            img = img_nparray.reshape(1, hight * width)[0]
            x_train[cnt] = img
            y_label[cnt] = float(name)
            cnt += 1

def getSize(count):
    size = 0
    for i in range(count):
        name = str(i)
        size += len(os.listdir(r'data/'+name))
    print(size)
    return size

if __name__ == '__main__':
    size = getSize(9)
    x_train = np.empty((size+1, 96 * 96))
    y_label = np.empty(size+1)

    feature_name = np.arange(0,96*96,1)

    x_train[0] = feature_name
    y_label[0] = 10

    for i in range(0):
        print('Start Label:'+str(i))
        changeImg(str(i))


    print(x_train.shape)
    print(y_label.shape)

    np.savetxt('features.csv',x_train,delimiter=',')
    np.savetxt('labels.csv',y_label,delimiter=',')
