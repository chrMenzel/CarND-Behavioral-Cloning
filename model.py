import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D

lines = []

# load driving_log.csv
with open('./driving_data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# remove header from lines
lines = lines[1:]

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './driving_data/data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)

    # get steering measurement for this point in time
    measurement = float(line[3])
    measurements.append(measurement)

# Convert data to numpy arrays because that is the format Keras requires
X_train = np.array(images)
y_train = np.array(measurements)

# build the model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

"""
adding LeNet architecture: 2 convolutional layers with pooling layer
and 3 fully connected layers
"""

# Convolutional layer 1 with pooling layer
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
# Convolutional layer 2 with pooling layer
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

# 3 fully connected layers
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# we use mean squared error (mse) because this is a regression network
# and NOT a classification network
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
