import argparse
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from model import *
from utils import *
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Dropout, Flatten, Input, Reshape
from keras.models import Model, Input
from keras.utils import np_utils

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, required=True, help="name of model want to fine tune")
ap.add_argument("-de", "--deforst",action='append', default=[], help="deforst layers between 'start' to 'end' you need 2 value for this variable")
ap.add_argument("-n", "--num_class", type=int, required=True, help="num of class in last layers")
ap.add_argument("-d", "--data", type=str, required=True, help="directory of data")
ap.add_argument("-l", "--label", type=str, required=True, help="directory of label")
ap.add_argument("-lr", "--learning_rate", type=int, default=0.00001, help="learning rate")
ap.add_argument("-lo", "--loss_function", type=str, default='categorical_crossentropy', help="loss function")
args = vars(ap.parse_args())

input_shape = {
    'InceptionV3': (299,299,3),
    'ResNet50': (224,224,3),
    'Xception': (299,299,3),
    'DenseNet121': (224,224,3)
}

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

loss_function = {
    'categorical_crossentropy': 'categorical_crossentropy',
    'focal_loss': focal_loss
}

if __name__ == "__main__":
#### Load base model ####
    base_model = MyModel(args["model"])
    base_model.freeze()
#### Deforst layers ####
    if len(args["deforst"]) == 2:
        print("Deforst "+ args["model"]+" model from "+ args["deforst"][0]+" to " + args["deforst"][1])
        try:
            base_model.defrost(args["deforst"][0],args["deforst"][1])
        except OSError as error:
            print(error)
        
    else:
        print("Define deforst value error, Not deforst any layers")

#### Define model use to fine tune ####
    input = Input(shape=input_shape[args["model"]])
    x = base_model.model(input)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1000,activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512,activation="relu")(x)
    prediction = Dense(args['num_class'],activation="softmax")(x)

    model = Model(input=input,output=prediction)
    print("Architecture of model:\n")
    model.summary()

#### Load data ####
    print('Loading data...')
    try:
        with open(args["data"],"rb") as file:
            data =  pickle.load(file)
        with open(args["label"],"rb") as file:
            label = pickle.load(file)
    except OSError as error:
        print(error)
    
    data = convertTo3Channel(data)
    label = np_utils.to_categorical(label,args["num_class"])

    Xtrain,Ytrain = shuffle(data,label)
    Xtrain,Ytrain,Xval,Yval = getTrainValSet(Xtrain,Ytrain,9/10)

#### compile model ####
    adam = Adam(lr=args["learning_rate"])
    model.compile(optimizer=adam, loss=loss_function[args["loss_function"]], metrics=['accuracy'])

#### fit model ####
    model_checkpoint = ModelCheckpoint(filepath='models/det-'+str(args["deforst"])+'_'+ args['model'] +'_'+str(args["learning_rate"])+'-sliceX.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=1)

    csv_logger = CSVLogger(filename='LNDb_model_v1.1_sliceX',
                        separator=',',
                        append=True)

    early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=0.0,
                                patience=10,
                                verbose=1)

    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.2,
                                            patience=8,
                                            verbose=1,
                                            epsilon=0.001,
                                            cooldown=0,
                                            min_lr=0.0000001)

    callbacks = [model_checkpoint,
                csv_logger,
                early_stopping,
                reduce_learning_rate]

    history = model.fit(Xtrain[:,0]/1000, Ytrain, batch_size=32, epochs=100, validation_data=[Xval[:,0]/1000,Yval], callbacks=callbacks)    

    
    plt.style.use('ggplot')
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["acc"], label="train_acc")
    plt.plot(history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig('png/det-'+str(args["deforst"])+'_'+args['model']+'_'+str(args["learning_rate"])+'-SliceX.png')
    plt.show()