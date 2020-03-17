import re
import pickle
import argparse
import numpy as np
import itertools
import matplotlib.pyplot as plt
from keras.models import load_model 
from utils import convertTo3Channel
from sklearn.metrics import classification_report, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="name of model want to load")
ap.add_argument("-d", "--data", type=str, required=True, help="directory of data")
ap.add_argument("-l", "--label", type=str, required=True, help="directory of label")
ap.add_argument("-lo", "--loss", type=str, default="categorical_crossentropy",help="loss function")
args = vars(ap.parse_args())

def draw_heatmap_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)

    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, classes, rotation = 90)
    plt.yticks(tick_marks, classes)

    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    thresol = cm.max()*0.7 + cm.min()*0.3
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[0])):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresol else "black")
    plt.colorbar()

    plt.title("Confusion matrix")
    plt.ylabel("True lable", rotation = 90)
    plt.xlabel("Predict lable")
    model_name = re.split("/",args["model"])
    plt.savefig("png/"+model_name[1]+"_Confusion-matrix.png")

if __name__ == "__main__":
#### load model ####
    model = load_model(args["model"])

#### load data ####
    with open(args["data"],"rb") as file:
        testSet = pickle.load(file)
    with open(args["label"],"rb") as file:
        label = pickle.load(file)

    testSet = convertTo3Channel(testSet)

#### prediction ####
    test = testSet[:,0]/1000
    y_pred = model.predict(test)

    y_true = label
    target_labels = ["GGO","Disease"]
    y_pred = np.argmax(y_pred, axis=1)

    draw_heatmap_confusion_matrix(y_true, y_pred, target_labels) 
    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=target_labels))
