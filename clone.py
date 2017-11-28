import csv
import cv2
import numpy as np
#import tqdm
#import sys
#import time
#import pyprind
from sklearn.model_selection import train_test_split

#instantiate progress bar object
#n = 100
#bar = pyprind.ProgBar(n)

lines = []
with open('../mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import sklearn
from matplotlib import pyplot
import matplotlib.pyplot as plt

#to do: later add left and right images
#to do later: add flipped images
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:


                #read in all 3 camera images
                center_image = cv2.imread('../mydata/IMG/' + batch_sample[0].split('\\')[-1])
                left_image = cv2.imread('../mydata/IMG/' + batch_sample[1].split('\\')[-1])
                right_image = cv2.imread('../mydata/IMG/' + batch_sample[2].split('\\')[-1])

                #read in the center steering angle placed in the 4th column in the csv file (D column)
                center_angle = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.2  # this is a parameter to tune
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                # add images and angles to data set
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)


                #filename = '../mydata/IMG/'+batch_sample[0].split('\\')[-1]


                #center_image = cv2.imread(filename)
                #center_angle = float(batch_sample[3])

                #w,h,ch = center_image.shape
                # cv2.imshow("cropped", center_image)
                # center_image.crop((0,25,w,h-70)).save(...)
                #images.append(center_image[65:h-25, 0:w])

                #images.append(center_image)
                #angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# images = []
# measurements = []
# for line in lines:
#     #loop through left, center, right images
#     for i in range(3):
#         source_path = line[i]
#         filename = source_path.split('/')[-1]
#         current_path = '../data/IMG/' + filename
#         image = cv2.imread(current_path)
#         images.append(image)
#         measurement = float(line[3])
#         measurements.append(measurement)

# aug_images, aug_measurements = [], []
# for image,measurement in zip(images,measurements):
#     aug_images.append(image)
#     aug_measurements.append(measurement)
#     aug_images.append(cv2.flip(image,1))
#     aug_measurements.append(measurement*-1.0)

#create giant train numpy arrays using the augmented images and augmented measurements
# X_train = np.array(aug_images)
# y_train = np.array(aug_measurements)

#import keras modules
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

#define convolution network
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()