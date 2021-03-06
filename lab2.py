import os
import numpy as np
import tensorflow as tf
import random
import sys
import argparse
import seaborn as sn
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from math import e, floor, sqrt
from statistics import mean

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ALGORITHM = "guesser"
# ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

# DATASET = "mnist_d"
# DATASET = "mnist_f"
# DATASET = "cifar_10"
DATASET = "cifar_100_f"
# DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 32**2
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 32**2                                
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 20
    IH = 32
    IW = 32
    IZ = 3
    IS = 32**2                               


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 10, layers=[64], batch_size=64,dropout = False, dropRate = 0.2):
    # print(x[-].shape,y[0].shape)
    tf_nn = keras.Sequential()
    # tf_nn.add(keras.layers.Flatten(input_shape=x[0].shape))
    for num in layers:
        tf_nn.add(keras.layers.Dense(num,activation='relu'))
    if (dropout):
        tf_nn.add(keras.layers.Dropout(dropRate))
    
    tf_nn.add(keras.layers.Dense(y.shape[1],activation='softmax'))

    tf_nn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    tf_nn.fit(x,y,batch_size=batch_size,epochs=eps)
    return tf_nn


def buildTFConvNet(x, y, eps = 20, hidden_layers=[256],batch_size=64,dropout = True, dropRate = 0.25,convPoolSeq=2,convLayer=[32,64],convSize=[5,5],poolSize=[2,2], saveCheckpoints=False,savePath=None,useVGG=False,validation_data=None,dataAug=False):

    print((x.shape[1],x.shape[2],x.shape[3]),y.shape[1])
    if (saveCheckpoints):
        print("Saving Weights during Training: Enabled")
        cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=savePath,save_weights_only=False,save_best_only=True,verbose=1)]
    else:
        print("Saving Weights during Training: Disabled")
        cp_callback = None
    if (not useVGG):
        if (os.path.exists(savePath)):
            print("found checkpoint for mnist, loading weights...")
            tf_conv_nn = keras.models.load_model(savePath)
            return tf_conv_nn
        else:
            print("Saved Model Not found, training from beginning...")
            tf_conv_nn = keras.Sequential()
            for i in range(convPoolSeq):
                if (i == 0):
                    tf_conv_nn.add(keras.layers.Conv2D(convLayer[i],convSize[i],activation='relu',input_shape=(x.shape[1],x.shape[2],x.shape[3])))
                else:
                    tf_conv_nn.add(keras.layers.Conv2D(convLayer[i],convSize[i],activation='relu'))
                tf_conv_nn.add(keras.layers.Conv2D(convLayer[i],convSize[i],activation='relu'))
                tf_conv_nn.add(keras.layers.MaxPool2D(pool_size=poolSize[i],strides=(2,2)))
                if (dropout):
                    tf_conv_nn.add(keras.layers.Dropout(dropRate))

            tf_conv_nn.add(keras.layers.Flatten())
            for num in hidden_layers:
                tf_conv_nn.add(keras.layers.Dense(num, activation='relu'))
            tf_conv_nn.add(keras.layers.BatchNormalization())
            # if (dropout):
            tf_conv_nn.add(keras.layers.Dropout(0.5))
            tf_conv_nn.add(keras.layers.Dense(y.shape[1], activation='softmax'))

            tf_conv_nn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        tf_conv_nn.fit(x,y,batch_size=batch_size,epochs=eps,callbacks=cp_callback)
        return tf_conv_nn
    else:
        if (os.path.exists(savePath)):
            print("found checkpoint for vgg, loading weights...")
            tf_vgg = keras.models.load_model(savePath)
            return tf_vgg
        else:
            print("Saved Model Not found, training from beginning...")
            tf_vgg = keras.Sequential()
            tf_vgg.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',kernel_initializer=keras.initializers.GlorotUniform(),padding='same',input_shape=(x.shape[1],x.shape[2],x.shape[3])))
            tf_vgg.add(keras.layers.MaxPool2D(pool_size=(2,2),padding='same',strides=2))
            tf_vgg.add(keras.layers.Dropout(0.5))
            
            tf_vgg.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',kernel_initializer=keras.initializers.GlorotUniform(),padding='same'))
            # tf_vgg.add(keras.layers.Conv2D(64,3,activation='relu',kernel_initializer='he_uniform'))
            tf_vgg.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',kernel_initializer=keras.initializers.GlorotUniform(),padding='same'))
            tf_vgg.add(keras.layers.MaxPool2D(pool_size=(2,2),padding='same',strides=2))
            tf_vgg.add(keras.layers.Dropout(0.5))

            tf_vgg.add(keras.layers.Flatten())
            tf_vgg.add(keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'))
            tf_vgg.add(keras.layers.Dropout(0.5))
            tf_vgg.add(keras.layers.Dense(y.shape[1], activation='softmax'))
            
            tf_vgg.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        if (dataAug):
            print("Utilizing Data Augmentation...")
            data_augmentor = keras.preprocessing.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
            iterator = data_augmentor.flow(x,y,batch_size=batch_size)
            steps = int(x.shape[0]/batch_size)
            tf_vgg.fit_generator(iterator,steps_per_epoch=steps,epochs=eps,callbacks=cp_callback,validation_data=validation_data)
        else:
            tf_vgg.fit(x,y,batch_size=batch_size,epochs=eps,callbacks=cp_callback,validation_data=validation_data)
        
        return tf_vgg
#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        cifar10 = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    elif DATASET == "cifar_100_f":
        cifar100_f = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar100_f.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        cifar100_c = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar100_c.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS*IZ))
        xTestP = xTest.reshape((xTest.shape[0], IS*IZ))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    
    # Normalize pixel values between 0-1
    xTrainP = np.divide(xTrainP,255)
    xTestP = np.divide(xTestP, 255)

    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data,hyperParams=None):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        if hyperParams:
            if (hyperParams["useVGG"]):
                return buildTFConvNet(xTrain,yTrain,eps=hyperParams["eps"], batch_size=hyperParams["batch_size"],
                                    saveCheckpoints=hyperParams["saveCheckpoints"],
                                    savePath=hyperParams["savePath"],useVGG=hyperParams["useVGG"],validation_data=hyperParams["validation_data"])
            else:
                return buildTFConvNet(xTrain,yTrain,eps=hyperParams["eps"],hidden_layers=hyperParams["hidden_layers"],
                                        batch_size=hyperParams["batch_size"],dropout=hyperParams["dropout"],
                                        dropRate=hyperParams["dropRate"],convPoolSeq=hyperParams["convPoolSeq"],
                                        convLayer=hyperParams["convLayer"],convSize=hyperParams["convSize"],
                                        poolSize=hyperParams["poolSize"],saveCheckpoints=hyperParams["saveCheckpoints"],
                                        savePath=hyperParams["savePath"])     
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")


def generate_confusion_matrix(truth,preds,plot=False):
    from sklearn import metrics

    confusion_matrix = metrics.confusion_matrix(truth.argmax(axis=1), preds.argmax(axis=1))

    print(confusion_matrix)
    print(metrics.classification_report(truth.argmax(axis=1), preds.argmax(axis=1)))

    if plot:
        # metrics.plot_confusion_matrix(truth.argmax(axis=1), preds.argmax(axis=1))
        sn.heatmap(confusion_matrix, annot=True)
        plt.show()
    return

def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    generate_confusion_matrix(yTest,preds,plot=True)

def plot_results(modelType='cnn',useResults=None):
    if (modelType=='ann'):
        useResults=[0.9786,0.8721,0.3749,0.1997,0.01]
        plt.bar(["mnist_d","mnist_f","cifar_10","cifar_coarse","cifar_fine"],useResults)
        plt.title("ANN Accuracies")
        plt.xlabel("Dataset")
        plt.ylabel("Accuracy")
        plt.show()
    elif (modelType=='cnn'):
        useResults=[0.9935,0.9206,0.7168,0.5391,0.4772]
        plt.bar(["mnist_d","mnist_f","cifar_10","cifar_coarse","cifar_fine"],useResults)
        plt.title("CNN Accuracies")
        plt.xlabel("Dataset")
        plt.ylabel("Accuracy")
        plt.show()
    else:
        print("Model Type not recognized")

#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    if ("cifar" in DATASET):
        hyperParams = {
            "useVGG": True,
            "dataAug": False,
            "eps": 100,
            "batch_size": 128,
            "saveCheckpoints": True,
            "validation_data":(data[1][0],data[1][1])
        }
        if (DATASET == "cifar_10"):
            hyperParams["savePath"] = "./saved_models/vgg_cifar10/vgg_cifar10.h5"
        elif (DATASET == "cifar_100_f"):
            if (hyperParams["dataAug"]):
                hyperParams["savePath"] = "./saved_models/vgg_cifar100f_aug/vgg_cifar100f_aug.h5"
            else:
                hyperParams["savePath"] = "./saved_models/vgg_cifar100f/vgg_cifar100f.h5"
        else:
            hyperParams["savePath"] = "./saved_models/vgg_cifar100c/vgg_cifar100c.h5"
        model = trainModel(data[0],hyperParams)
    else:
        hyperParams = {
            "useVGG": False,
            "dataAug": False,
            "eps": 20,
            "hidden_layers": [256],
            "batch_size": 64,
            "dropout": True,
            "dropRate": 0.25,
            "convPoolSeq": 2,
            "convLayer": [32,64],
            "convSize": [5,5],
            "poolSize": [2,2],
            "saveCheckpoints": True,
            "savePath": "./saved_models/cnn_mnist_d/cnn_mnist_d.h5",
            "validation_data":(data[1][0],data[1][1])
        }
        if (DATASET == "mnist_f"):
            hyperParams["savePath"] = "./saved_models/cnn_mnist_f/cnn_mnist_f.h5"
        model = trainModel(data[0],hyperParams)
    # model = test_func(data[0][0],data[0][1])
    # model = keras.models.load_model("./saved_models/vgg/vgg.h5")
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)
    # plot_results(modelType='cnn')


if __name__ == '__main__':
    main()