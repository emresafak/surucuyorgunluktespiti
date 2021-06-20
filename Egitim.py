
#Gerekli kütüphaneler import edilir

import tensorflow as tf
import tensorflow.keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
from keras.preprocessing.image import ImageDataGenerator
import cv2



img_gen = ImageDataGenerator(validation_split=0.2,rescale=1/255)

# Eğitim için veri seti hazırlanıyor

train_path = 'C://Users/EMRE/Desktop/dataset/train'
valid_path = 'C://Users/EMRE/Desktop/dataset/valid'
test_path = 'C://Users/EMRE/Desktop/dataset/test'


train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
    train_path, target_size=(224,224), batch_size=1024)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
    valid_path, target_size=(224,224), batch_size=256)
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
    test_path, target_size=(224,224), batch_size=256, shuffle=False)
	
	

#Hiper parametreler

NUMBER_OF_EPOCHS = 30
BATCH_SIZE = 32
STEPS_PER_EPOCH = len(X_train) / 32
STEPS_PER_VALIDATION = len(X_val) / 32

verbosity = 1
loss_function = 'binary_crossentropy'
learning_rate = 0.001
threshold = 0.5


#Evrişimsel Snir Ağı mimarisi

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding = 'same', activation='relu', input_shape=(height, width, dims)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3,3), padding= 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('sigmoid'))


model.compile(optimizer = Adam(lr = learning_rate), loss = loss_function, metrics = ['accuracy'])

#Model eğitim işlemi

model.fit(train_batches, steps_per_epoch = STEPS_PER_EPOCH, 
                    validation_steps = STEPS_PER_VALIDATION, valid_path,
                    epochs = NUMBER_OF_EPOCHS)
					
					

#Model değerlendirme işlemi
					
loss, acc = model.evaluate(test_batches, batch_size = BATCH_SIZE, verbose = verbosity)


print(loss)
print(acc)




model.save("model.h5")




