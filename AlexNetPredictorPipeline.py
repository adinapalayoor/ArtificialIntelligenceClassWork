# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:40:44 2023

@author: adina_l1uzsjt
"""

import cv2
import numpy as np
import os
import matplotlib as plt
#from torchvision.utils import make_grid
import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import Xception
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy


def load_images(folder_path, num_images=70, target_size=(224, 224)):
    #Initialize an empty list to store images
    images = []

    #Get a list of all image files in the folder
    image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]

    #randomly choose subset of images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    #loop through the selected files
    for filename in selected_images:
        #Construct the full file path
        file_path = os.path.join(folder_path, filename)
        #Read the image using OpenCV
        image = cv2.imread(file_path)
        #Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        #Append the image to the list
        images.append(image)

    #Convert the list of images to a NumPy array
    images_array = np.array(images, dtype=object)

    return images_array

    
def train_test_images(data, test_size=0.28):
    #split dataset into training an test
    train_images, test_images = train_test_split(data, test_size=test_size, random_state=42)
    return train_images, test_images

def assign_data_labels(melanoma_images,naevus_images):
    
    melanoma_data = melanoma_images
    naevus_data = naevus_images
    #assign a zero to melanoma images
    #assign a one to naevus images
    melanoma_labels= np.zeros(len(melanoma_data))
    naevus_labels= np.ones(len(melanoma_data))
    #combine data into one frame
    all_data = np.concatenate([melanoma_data, naevus_data], axis=0)
    all_labels = np.concatenate([melanoma_labels, naevus_labels], axis=0)
    
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    return all_data, all_labels

def AlexNet(input_shape=(224, 224, 3), num_classes=2,dropout_rate=0.5):
    # Initialize the model
    model = Sequential()

    # Layer 1: Convolutional layer + Max-pooling layer
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # Layer 2: Convolutional layer + Max-pooling layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # Layers 3-5: Three convolutional layers + 1 Max-pooling layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # Layers 6 - 8: Two fully connected hidden layers and one fully connected output layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))  # Specify dropout rate here
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))  # Specify dropout rate here
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with a loss function, a metric, and an optimizer method for estimating the loss function
    opt = SGD(lr=0.04)
    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def main():
    
    melanoma_path = r"complete_mednode_dataset\melanoma"
    naevus_path = r"complete_mednode_dataset\naevus"
    melanoma_images = load_images(melanoma_path)
    naevus_images = load_images(naevus_path, num_images=70)
    print(f"Length of melanoma images array:{len(melanoma_images)}")
    print(f"Length of naevus images array:{len(naevus_images)}")
    
    train_images_m, test_images_m = train_test_images(melanoma_images)
    print(f"Length of melanoma images array for training:{len(train_images_m)}")
    print(f"Length of melanoma images array for test:{len(test_images_m)}")
    
    train_images_n, test_images_n = train_test_images(naevus_images)
    print(f"Length of naevus images array for training:{len(train_images_n)}")
    print(f"Length of naevus images array for test:{len(test_images_n)}")
    
    print("Assigning all training data to a label:")
    
    data_training, data_training_labels = assign_data_labels(train_images_m, train_images_n)
    print(f"Length of full training data: {len(data_training)}")
    data_test, data_test_labels = assign_data_labels(test_images_m, test_images_n)
    print(f"Length of full test data: {len(data_test)}")
    

    data_training = data_training.astype('float32')
    data_test = data_test.astype('float32')
    data_training_labels_onehot = to_categorical(data_training_labels, dtype='float32')
    data_test_labels_onehot = to_categorical(data_test_labels, dtype='float32')
    
    best_dropout=0
    training_accuracy_best=0
    dropout_rates = [0.1,0.2,0.3, 0.4,0.5, 0.6, 0.7, 0.8,0.9]
    
    for dropout_rate in dropout_rates:
        alexnet_model = AlexNet(dropout_rate=dropout_rate)
    
        #Train the model on the training data
        batch_size = 32
        num_epochs = 10

        alexnet_model.fit(data_training, data_training_labels_onehot, batch_size=batch_size, epochs=num_epochs, validation_split=0.1)
        
        training_loss, training_accuracy = alexnet_model.evaluate(data_training, data_training_labels_onehot)
        print(f"Training Accuracy with Dropout Rate {dropout_rate}: {training_accuracy * 100:.2f}%")
        
        if training_accuracy>training_accuracy_best:
            best_dropout=dropout_rate
            training_accuracy_best = training_accuracy
            print(f"This is the new best dropout_rate:{dropout_rate}")
            
    print(f"The best dropout rate from training data is {best_dropout} with a image classification accuracy of {training_accuracy_best} ")
    #evaluate model on best dropout rate
    alexnet_model = AlexNet(dropout_rate=best_dropout)
    batch_size = 32
    num_epochs = 10

    alexnet_model.fit(data_training, data_training_labels_onehot, batch_size=batch_size, epochs=num_epochs, validation_split=0.1)
    test_loss, test_accuracy = alexnet_model.evaluate(data_test, data_test_labels_onehot)
    
    print(f"Test Accuracy with Dropout Rate {best_dropout}: {test_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()