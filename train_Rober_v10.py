#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:22:17 2017

@author: Rober
"""

import csv
import cv2
import numpy as np

rows = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for row in reader:
        rows.append(row)
        
images = []
steering = []
for row in rows:
    
    source_path = row[0]
    filename = source_path.split('/')[-1]
    current_path= './data/IMG/' + filename
    
    center_angle = float(row[3])
    abs_center_angle = abs(center_angle)
    # Only high steering values
    if abs_center_angle > 0.1:
        image = cv2.imread(current_path)
        images.append(image)
        steer_value = float(row[3])
        steering.append(steer_value)
    
    '''
    steering_center = float(row[3])
    
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    path = "./data/" # fill in the path to your training IMG directory
    img_center = process_image(np.asarray(Image.open(path + row[0])))
    img_left = process_image(np.asarray(Image.open(path + row[1])))
    img_right = process_image(np.asarray(Image.open(path + row[2])))

    # add images and angles to data set
    car_images.extend(img_center, img_left, img_right)
    steering_angles.extend(steering_center, steering_left, steering_right)
            
    image = cv2.imread(current_path)
    images.append(image)
    steering.append(steering_center)
    '''
    
###############################################
#Data augmentation
#DS is not-balanced. 
#DS has many values <0 
#=> steering to left => Because circuit is a counterclockwise [CCW] loop.    
augmented_images, augmented_steering = [],[]
for img,steer in zip(images, steering):
    augmented_images.append(img)
    augmented_steering.append(steer)
    augmented_images.append(cv2.flip(img, 1))
    augmented_steering.append(steer * -1.0)
    augmented_images.append(cv2.flip(img, 0))
    augmented_steering.append(steer * 1.0)



X = np.array(augmented_images)
Y = np.array(augmented_steering)
###############################################

###############################################
#Check shapes
print(X.shape)
print(Y.shape)

###############################################
#Shuffle ds, split in train, valid & test
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

X, Y = shuffle(X, Y)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y,  test_size=0.1)



###############################################
#Check shapes, lens, 
n_train = len(X_train)
n_valid = len(X_valid)
image_shape = X_train[0].shape


assert(len(X_train) == len(Y_train))
assert(len(X_valid) == len(Y_valid))

print("Number of training examples =", n_train)
print("Number of valid examples =", n_valid)
print(X_train.shape)
print(Y_train.shape)

##########################
#Model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D
#from keras import losses

#Init Variables
batch_size = 128
epochs = 20
do = [1.,1.,1.,1.]
learning_rate=0.001
# input image dimensions
img_rows, img_cols = 160, 320
input_shape = (img_rows, img_cols, 3)

###############################################
#Data Augmentation Generator
#https://keras.io/preprocessing/image/
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#ZCA whitening
#http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        samplewise_center=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False,
        rescale=1.)
###############################################

#Define Model

model = Sequential()
#CNN1
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24, kernel_size=(5, 5),strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5),strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(48, kernel_size=(5, 5),strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
#Flatten
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
#Output
model.add(Dense(1))

model.summary()


#LOAD PRETRAINED MODEL
pretrained_model = True
if pretrained_model:
    print('Loading Pretrained model')
    model.load_weights('./cp/model-p3-v08.h5')



from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpointer = ModelCheckpoint(filepath="./cp/model-p3-v10.h5", verbose=1, save_best_only=True,
                              monitor='val_loss', save_weights_only=False, mode='auto')

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')


model.compile(loss='mse', 
              optimizer=keras.optimizers.Adam(lr=learning_rate), 
              metrics=None, 
              sample_weight_mode=None)

history_object = model.fit_generator(
    datagen.flow(X_train, Y_train, batch_size=batch_size),
    steps_per_epoch= len(X_train)//batch_size,
    nb_epoch=epochs,
    verbose=1,
    validation_data=datagen.flow(X_valid,Y_valid,batch_size=batch_size),
    validation_steps=len(X_valid) // batch_size,
    callbacks=[checkpointer,earlystopping],
    initial_epoch=0)

##############################################################
import matplotlib.pyplot as plt
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

fig = plt.figure()
fig.savefig('model-p3-v08.png', dpi=fig.dpi)








