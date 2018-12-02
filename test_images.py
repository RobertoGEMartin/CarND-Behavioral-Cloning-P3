#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:00:31 2017

@author: Rober
"""

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read():
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
        filter_ = ( center_angle > 0.01) | (center_angle < 0 )
        if (True):
            image = cv2.imread(current_path)
            images.append(image)
            steer_value = float(row[3])
            steering.append(steer_value)
    return np.array(images),  np.array(steering)

def select_idx(x, a, b):
    select_indices = np.where(np.logical_or( x < a , x > b ))
    return select_indices
    
def hist(x):
    import matplotlib.pyplot as plt
    # the histogram of the data
    plt.hist(x, 100)
    
    plt.xlabel('Values')
    plt.ylabel('#Samples')
    plt.title(r'Steering Values')
    plt.grid(True)
    plt.show()
    print(x.shape)


def flip_h_v(): 
    rows = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for row in reader:
            rows.append(row)       
            
    for i,row in enumerate(rows):
        
        source_path = row[0]
        filename = source_path.split('/')[-1]
        current_path= './data/IMG/' + filename
        img=cv2.imread(current_path)
        rimg=img.copy()
        fimg=img.copy()
        rimg=cv2.flip(img,1)
        fimg=cv2.flip(img,0)
        cv2.imshow("Original", img)
        cv2.imshow("horizontal flip", rimg)
        cv2.imshow("vertical flip", fimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if i>1:
            break
    
def save_disk_example():    
    fig = plt.figure()
    plt.plot(np.random.uniform(-1,0,20))
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    fig.savefig('model-p3-v09.png', dpi=fig.dpi)


def get_images():
    #@behnamprime https://github.com/behnamprime/UDACITY-CarND-Behavioral-Cloning-P3
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    center_img = mpimg.imread('./data/IMG/center_2016_12_01_13_39_58_284.jpg')
    left_img = mpimg.imread('./data/IMG/left_2016_12_01_13_39_58_284.jpg')
    right_img = mpimg.imread('./data/IMG/right_2016_12_01_13_39_58_284.jpg')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(left_img)
    ax1.set_title('Left Camera', fontsize=9)
    ax2.imshow(center_img)
    ax2.set_title('Center Camera', fontsize=9)
    ax3.imshow(right_img)
    ax3.set_title('Right Camera', fontsize=9)
    plt.subplots_adjust(left=0., right=1, top=2, bottom=0.)

#get_images()

images,  steering = read()


hist(np.array(steering))

idx = select_idx(steering, a=0,b=0.003 )
new_steering = steering[idx]
new_images = images[idx]
hist(np.array(new_steering))

#mu, sigma = 0, 0.1
#x = np.random.normal(mu, sigma,size=1000)
#hist(x)
#bins = 10
#max_ = np.max(x)
#min_ = np.min(x)
#for i in range(bins):
#    step = (max_ - min_)/bins
#    a = i * step
#    b = a + step
#    idx = select_idx(x, a=a,b=b )
#    x = new_steering[idx]
#    hist(x)



