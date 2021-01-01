import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # to avoid tf warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np


"""
    Function name : load_data
    Purpose : To read the datasets required
"""
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # change datatype to float
    x_train.astype('float32')
    x_test.astype('float32')
    # bring values to [0,1] range
    x_train = x_train/255.0
    x_test = x_test/255.0
    # reshape the data to required shape
    x_train = x_train.reshape((x_train.shape[0],28,28,1))
    x_test = x_test.reshape((x_test.shape[0],28,28,1))
    # one-hot encoding of target classes
    y_train = keras.utils.to_categorical(y_train,num_classes=10)
    y_test = keras.utils.to_categorical(y_test,num_classes=10)
    # return all obtained arrays
    return x_train,y_train,x_test,y_test


"""
    Function name : main
    Purpose : To build a model, train and test it.
"""
def main():
    # use load_data function to get required dataset
    x_train, y_train, x_test, y_test = load_data()
    # model
    model = keras.models.Sequential()   # Sequential model in keras
    model.add(layers.Conv2D(128,activation='relu',kernel_size=(5,5),input_shape=(28,28,1))) # 2D Conv layer
    model.add(layers.MaxPooling2D(pool_size=(2,2))) # Max pooling layer
    model.add(layers.Conv2D(32,activation='relu',kernel_size=(5,5)))    # 2D Conv layer
    model.add(layers.MaxPooling2D(pool_size=(2,2))) # Max pooling layer
    model.add(layers.Flatten()) # flatten the matrices
    model.add(layers.Dense(units=32,activation='relu')) # Fully connected layer 
    model.add(layers.Dense(units=10,activation='softmax'))  # Fully connected output layer
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])    # compile the model
    model.summary()     # print model summary
    print()
    batch_size = int(input("Batch size = "))    # batch size input
    epochs = int(input("Number of epochs = "))  # number of epochs input
    print("\nTraining started ... \n")
    history = model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs)    # train the model on given data
    print("\nTraining completed !\n")
    print("Training summary : Error = {:.4f}\tAccuracy = {:.2f}%\n".format(history.history['loss'][-1],history.history['accuracy'][-1]*100))
    print("Testing the trained model on a test dataset ...\n")
    scores = model.evaluate(x=x_test,y=y_test,batch_size=batch_size)    # test the model on given data
    print("\nTesting results  : Error = {:.4f}\tAccuracy = {:.2f}%\n".format(scores[0],scores[1]*100))

    save = input("Do you want to the save the model? (y/n) : ")
    if save.lower() == 'y':
        model.save("digit.pb")


if __name__ == "__main__":
    main()
