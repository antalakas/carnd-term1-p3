import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Cropping2D, SpatialDropout2D, Dropout, ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# input folders
pathTrack1Forth4 = 'C:\\Users\\antal\\udacity\\self-driving-car-sim\Data\\Track1Forth4\\'
pathTrack1Forth = 'C:\\Users\\antal\\udacity\\self-driving-car-sim\Data\\Track1Forth\\'
pathTrack1Back = 'C:\\Users\\antal\\udacity\\self-driving-car-sim\Data\\Track1Back\\'
pathTrack1Additional = 'C:\\Users\\antal\\udacity\\self-driving-car-sim\Data\\Track1Additional\\'
PlayedByAndreasWithAdditionalTurns = \
    'C:\\Users\\antal\\udacity\\self-driving-car-sim\Data\\PlayedByAndreasWithAdditionalTurns\\'

pathTrack2Forth = 'C:\\Users\\antal\\udacity\\self-driving-car-sim\Data\\Track2Forth\\'

# lines in csv file
samples = []

# function to read lines from multiple csv files and concatenate them in a single array
# depending on the arguments the csv file is enriched with additional columns in order to enable
# processing later during training
def read_csv(path, remove_low_measurement=False, flip=False, need_augment = False):

    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:

            # if decided we can drop images having small angle measurements
            if remove_low_measurement:
                if abs(float(line[3])) < 0.001:
                    if np.random.rand() > 0.33:
                        continue

            for i in range(3):
                filename = line[i].split('\\')[-1]
                line[i] = path + "IMG\\" + filename

            # 7 - original
            line.append(0)
            # 8 - flip
            line.append(0)
            # 9 - augment
            line.append(0)
            # 10 - flip - augment
            line.append(0)

            original_line = line[:]
            original_line[7] = 1
            samples.append(original_line)

            if flip:
                new_line = line[:]
                new_line[8] = 1
                samples.append(new_line)

            if need_augment and np.random.rand() < 0.5:
                new_line = line[:]
                new_line[9] = 1
                samples.append(new_line)

            if flip and need_augment and np.random.rand() < 0.5:
                new_line = line[:]
                new_line[10] = 1
                samples.append(new_line)

# for a successful Track 1 recording we used driving data from
# four (4) forward rounds and some additional images from tha part with successive turns
#  All images are flipped

# remove_low_measurement, flip, augment
read_csv(pathTrack1Forth4, False, True, False)
# read_csv(pathTrack2Forth, False, True, True)
# read_csv(pathTrack1Forth, False, False, True)
# read_csv(pathTrack1Back, False, False, True)
read_csv(pathTrack1Additional, False, True, False)
# read_csv(PlayedByAndreasWithAdditionalTurns, False, False, True)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# function to randomly translate image
def translate_image(image, steer, trans_range = 100):
    tr_x = trans_range*np.random.uniform()-trans_range/2
    compensation = tr_x / trans_range * .2
    steer_ang = steer + compensation
    tr_y = 40 * np.random.uniform() - 40 / 2
    trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, trans_m, (320, 160))

    return image_tr, steer_ang

# generate image with random brightness
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5+np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2]*random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

    return image1

# generate image with random shadow
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >= 0)] = 1

    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1]*random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0]*random_bright
    image1 = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return image1

# function to apply augmentation
def augment(center_image, center_angle, left_image, left_angle, right_image, right_angle):
    # # translate
    # if np.random.rand() < 0.33:
    #     center_image, center_angle = translate_image(center_image, center_angle)
    #     left_image, left_angle = translate_image(left_image, left_angle)
    #     right_image, right_angle = translate_image(right_image, right_angle)

    # brightness
    center_image = augment_brightness_camera_images(center_image)
    # shadow
    center_image = add_random_shadow(center_image)

    # brightness
    left_image = augment_brightness_camera_images(left_image)
    # shadow
    left_image = add_random_shadow(left_image)

    # brightness
    right_image = augment_brightness_camera_images(right_image)
    # shadow
    right_image = add_random_shadow(right_image)

    return center_image, center_angle, left_image, left_angle, right_image, right_angle

# generator that builds upon the enriched line data from csv file and applies training, validation
# to original images, flipped images and augmented images
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])

                left_image = cv2.imread(batch_sample[1])
                left_angle = center_angle + 0.2

                right_image = cv2.imread(batch_sample[2])
                right_angle = center_angle - 0.2

                # original
                if batch_sample[7] == 1:
                    images.append(center_image)
                    angles.append(center_angle)
                    images.append(left_image)
                    angles.append(left_angle)
                    images.append(right_image)
                    angles.append(right_angle)

                # flip
                if batch_sample[8] == 1:
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)
                    images.append(np.fliplr(left_image))
                    angles.append(-left_angle)
                    images.append(np.fliplr(right_image))
                    angles.append(-right_angle)

                # augment
                if batch_sample[9] == 1:
                        center_image, center_angle, left_image, left_angle, right_image, right_angle = \
                            augment(center_image, center_angle, left_image, left_angle, right_image, right_angle)

                        images.append(center_image)
                        angles.append(center_angle)

                        images.append(left_image)
                        angles.append(left_angle)

                        images.append(right_image)
                        angles.append(right_angle)

                # flip - augment
                if batch_sample[10] == 1:
                    center_image = np.fliplr(center_image)
                    center_angle = -center_angle
                    left_image = np.fliplr(left_image)
                    left_angle = -left_angle
                    right_image = np.fliplr(right_image)
                    right_angle = -right_angle

                    center_image, center_angle, left_image, left_angle, right_image, right_angle = \
                        augment(center_image, center_angle, left_image, left_angle, right_image, right_angle)

                    images.append(center_image)
                    angles.append(center_angle)

                    images.append(left_image)
                    angles.append(left_angle)

                    images.append(right_image)
                    angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.summary()

    return model

# the nvidia model with the addition of dropout after convolutional layers
def nvidia_model_single_dropout():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.summary()

    return model

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = nvidia_model_single_dropout()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=3 * len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=3 * len(validation_samples),
                                     nb_epoch=3)

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

model.save('model.h5')