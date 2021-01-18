import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

standard_path = '../../opt/carnd_p3/data/'
lines = []

# load driving_log.csv
with open(standard_path + 'driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# remove header from lines
lines = lines[1:]

print('There are ' + str(len(lines)) + ' lines in the csv file.')
images, measurements = [], []

correction = 0.3  # this is a parameter for the left and right images to tune the steering measurement

# A safety check if pictures exist
image = cv2.imread(standard_path + 'IMG/center_2016_12_01_13_30_48_287.jpg')
if image is None:
    print('image is null')
    exit()

print('start building measurements ...')

for line in lines:
    for i in range(3):
        # Load images from center and left/right camera
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = standard_path + 'IMG/' + filename
        image = cv2.imread(current_path)
        if image is None:
            print(current_path)
            print("image not found from line " + str(i))
            exit()
        else:
            images.append(image)

    # get steering measurements for this point in time
    measurement = float(line[3])
    measurements.append(measurement)
    # correct only if measurement not too near at zero
    if measurement >= -0.05 and measurement <= 0.05:
        measurements.append(measurement)
        measurements.append(measurement)
    else:
        measurements.append(measurement + correction)
        measurements.append(measurement - correction)

print('measurements built')

print('flip images starts ...')
# flip images horizontally to get more data for training
augmented_images, augmented_measurements = [], []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1)

print('flip images finished')

# Convert data to numpy arrays because that is the format Keras requires
X_train = np.array(augmented_images)
print('X_train ready')
y_train = np.array(augmented_measurements)
print('y_train ready')
print('converting ready')
# build the model
model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
"""
adding NVIDIA architecture
"""

# 5 Convolutional layers
model.add(Convolution2D(24, (5, 5), subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, (5, 5), subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, (5, 5), subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(Flatten())

# 4 fully connected dense layers with dropout layers between
model.add(Dense(100))
model.add(Dropout(.1))
model.add(Dense(50))
model.add(Dropout(.1))
model.add(Dense(10))
model.add(Dropout(.1))
model.add(Dense(1))

# we use mean squared error (mse) because this is a regression network
# and NOT a classification network
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
