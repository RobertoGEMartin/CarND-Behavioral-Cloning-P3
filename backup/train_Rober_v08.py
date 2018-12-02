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
    if abs_center_angle > 0.003:
        image = cv2.imread(current_path)
        images.append(image)
        steer_value = float(row[3])
        steering.append(steer_value)
    

    
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
batch_size = 32
epochs = 50
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
checkpointer = ModelCheckpoint(filepath="./cp/model-p3-v08.h5", verbose=1, save_best_only=True,
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
fig = plt.figure()

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

fig.savefig('model-p3-v08.png', dpi=fig.dpi)

##############################################################
##END
##############################################################

'''
##############################################################
#Functions: to test and to get hist & images
##############################################################
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

get_images()

#images,  steering = read()
#
#
#hist(np.array(steering))
#
#idx = select_idx(steering, a=0,b=0.01 )
#new_steering = steering[idx]
#new_images = images[idx]
#hist(np.array(new_steering))

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




