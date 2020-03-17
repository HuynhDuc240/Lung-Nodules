import keras.backend as K
import pickle
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Dropout, Flatten, Input, Reshape
from keras.models import Model, Input

from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121

from keras.utils import np_utils
import tensorflow as tf

dispatcher={'InceptionV3': InceptionV3,
            'ResNet50': ResNet50,
            'Xception': Xception,
            'DenseNet121': DenseNet121}

class MyModel():
    def __init__(self,name='InceptionV3'):
        self.name = name
        try:
            self.model = dispatcher[name](weights='imagenet',include_top=False)
            pass
        except OSError as error:
            print(error)
            pass
    def freeze(self):
        for l in self.model.layers:
            l.trainable = False

    def defrost(self, start, end):
        if type(start) == str and type(end) == str:
            start = self.getIndex(start)
            end = self.getIndex(end)
        for l in self.model.layers[start:end+1]:
            l.trainable = True

    def getIndex(self,layerName):
        index = None
        for idx, layer in enumerate(self.model.layers):
            # print(layer.name)
            if layer.name == layerName:
                index = idx
                break
        return index

    def summary(self):
        self.model.summary() 

if __name__ == "__main__":
    model = MyModel('InceptionV3')
    model.freeze()
    model.defrost('mixed9','mixed10')
    model.summary()
    pass
