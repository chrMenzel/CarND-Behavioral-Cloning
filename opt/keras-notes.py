
"""
The keras.models.Sequential class is a wrapper for the neural network model.
It provides common functions like fit(), evaluate(), and compile().
We'll cover these functions as we get to them. Let's start looking at the
layers of the model.

See the documentation for keras.models.Sequential in Keras 2.09 here.

https://faroit.com/keras-docs/2.0.9/models/sequential/

"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Create the Sequential model
model = Sequential()

"""
# Layers

A Keras layer is just like a neural network layer.
There are fully connected layers, max pool layers, and activation layers.
You can add a layer to the model using the model's add() function.
For example, a simple model would look like this:
"""

# 1st Layer - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 3)))

# 2nd Layer - Add a fully connected layer
model.add(Dense(128))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

# 4th Layer - Add a fully connected layer (Output width = 60)
model.add(Dense(60))

# 5th Layer - Add a ReLU activation layer
model.add(Activation('relu'))

# 7. Convolutions in Keras
model_convolutional = Sequential()
# Add a convolutional layer with 32 filters, a 3x3 kernel,
# and valid padding before the flatten layer.
model_convolutional.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model_convolutional.add(Activation('relu'))
model_convolutional.add(Flatten())
model_convolutional.add(Dense(128))
model_convolutional.add(Activation('relu'))
model_convolutional.add(Dense(5))
model_convolutional.add(Activation('softmax'))

# 8. Pooling in Keras
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

# 9. Dropout in Keras
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

# 10. Testing in Keras
