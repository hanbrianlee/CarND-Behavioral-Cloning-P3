import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

lines = []
#change mydata folder to data folder if udacity provided data needs to be used
with open('../mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#split train and validation samples
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import sklearn
from matplotlib import pyplot
import matplotlib.pyplot as plt

#save cropped image for report
test_image = cv2.imread('../mydata/IMG/' + lines[0][0].split('\\')[-1])
h,w,ch = test_image.shape
cv2.imwrite('cropimg.png', test_image[65:h-25, 0:w])

def generator(samples, batch_size=32):
    num_samples = len(samples) #using fit_generator, this line never gets run more than the 1st time it runs. Use of generator is indicated by yield line at the end of this while loop.
    while 1: # Loop forever so the generator never terminates
        #set up batches
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            #define frame image and steering angle arrays
            images = []
            angles = []
            for batch_sample in batch_samples:


                #read in all 3 camera images
                #change split('\\') to ('/') if udacity provided data is to be used. Windows simulator automatically adds \\ between directories
                center_image = cv2.imread('../mydata/IMG/' + batch_sample[0].split('\\')[-1])
                left_image = cv2.imread('../mydata/IMG/' + batch_sample[1].split('\\')[-1])
                right_image = cv2.imread('../mydata/IMG/' + batch_sample[2].split('\\')[-1])

                #read in the center steering angle placed in the 4th column in the csv file (D column)
                center_angle = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.2  #tunable parameter
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                # add images and angles to data set, could be later improved by using extend method
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            #put the images and angles arrays into np.array format
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#import keras modules
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
#from keras.regularizers import l2
#import pydot_ng as pydot
#from keras.utils.visualize_util import plot


#define convolution network (closely resembles nvidia deep learning network for self driving car
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
#model.add(Dropout(0.5)) #add a dropout layer to reduce overfitting
#dropout/pooling/regularization not applied since my model was not overfitting and worked fine without it
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#use adam optimizer
model.compile(loss='mse', optimizer='adam')
#store the fit result into history_object to be later visualized for diagnosis
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, verbose=1)

model.summary()
model.save('model.h5')

#plot(model, to_file='model.png')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('perfgraph.png', bbox_inches='tight')