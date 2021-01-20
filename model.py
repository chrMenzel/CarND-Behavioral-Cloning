import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D

standard_path = '../../opt/carnd_p3/data/'

def get_csv_lines():
    """
    Puts the lines of provided csv file 'driving_log.csv' in an array
    and returns this array
    """
    lines = []

    # load driving_log.csv
    with open(standard_path + 'driving_log.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    # remove header from lines
    return lines[1:]


def get_images_and_angles(lines):
    """
    Reads center, left and right camera image and corresponding
    steering angle. The steering angle for the left and right
    camera image is corrected if the steering angle is far
    enough away from zero.

    Returns 2 arrays with all images and corresponding (corrected)
    steering angles
    """
    # the images and steering angles are getting stored in arrays
    images, angles = [], []
    # this is a parameter for left/right images to tune the steering measurement
    correction = 0.3

    for line in lines:
        # get center, left and right camera image
        for i in range(3):
            # Load images from center and left/right camera
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = standard_path + 'IMG/' + filename
            image = cv2.imread(current_path)
            # Check if this image exists, if not exit
            if image is None:
                print(current_path)
                print("image not found from line " + str(line))
                exit()
            else:
                # image exists, as cv2 loads images in BGR format,
                # we have to convert it to RGB format
                imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(imageRGB)

        # get steering angle for the 3 images for this point in time
        angle = float(line[3])
        angles.append(angle)
        # correct only if measurement is not too near at zero
        if angle >= -0.05 and angle <= 0.05:
            angles.append(angle)
            angles.append(angle)
        else:
            # if angle is far away from zero, correct the
            # angle to avoid driving off the road
            angles.append(angle + correction)
            angles.append(angle - correction)
            
    return images, angles


def flip_images(images, angles):
    augmented_images, augmented_angles = [], []

    for image, angle in zip(images, angles):
        # put the original image and angle in array
        augmented_images.append(image)
        augmented_angles.append(angle)
        # flip the image and turn to RGB
        flipped_image = cv2.flip(image, 1)
        flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
        # add flipped image and inverted angle to array
        augmented_images.append(flipped_image)
        augmented_angles.append(angle * -1)

    return augmented_images, augmented_angles


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    """
    adding NVIDIA architecture
    """

    # 5 Convolutional layers
    model.add(Convolution2D(24, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, (5, 5), subsample=(2, 2), activation='relu'))
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

    return model


# read csv file
lines = get_csv_lines()
print('There are ' + str(len(lines)) + ' lines in the csv file.')

# check if images exist
image = cv2.imread(standard_path + 'IMG/center_2016_12_01_13_30_48_287.jpg')
if image is None:
    print('image is null')
    exit()

# load all images and steering angles
images, angles = get_images_and_angles(lines)
print(np.shape(images))
print(np.shape(angles))
# flip all images horizontally to get more data for training
images, angles = flip_images(images, angles)
print(np.shape(images))
print(np.shape(angles))

# Convert data to numpy arrays because that is the format Keras requires
X_train = np.array(images)
y_train = np.array(angles)

# build, compile, train and save the model
model = build_model()
# we use mean squared error (mse) because this is a regression network
# and NOT a classification network
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
