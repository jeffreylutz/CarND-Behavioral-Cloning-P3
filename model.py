'''
Trains model for determining steering angle
'''
import os

from keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D, Conv2D, ELU, Flatten, Dense, Dropout, Lambda, Reshape, MaxPooling2D

# fix random seed for reproducibility
np.random.seed(7)

model_dir = './models/'


def nvidia_model(modeljsonfile, dropout, lr):
    # in_row, in_col, in_depth = 66, 200, 3
    in_row, in_col, in_depth = 160, 320, 3

    model = Sequential()

    # normalize image values between -1.0 : 1.0
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(in_row, in_col, in_depth)))
    # model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(in_row, in_col, in_depth), output_shape=(in_row, in_col, in_depth)))

    ## horizon: 70, front of car: 25
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(in_row, in_col, in_depth)))

    # valid border mode should get rid of a couple each way, whereas same keeps
    # Use relu (non-linear activation function), not mentioned in Nvidia paper but a standard
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Dropout(rate=dropout))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Dropout(rate=dropout))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(rate=dropout))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(rate=dropout))
    model.add(Flatten())
    model.add(Dropout(rate=dropout))
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(rate=dropout))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    # compile with normal adam optimizer (loss .001) and return
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    # exit(1)
    with open(modeljsonfile, 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
    return model


def nvidia_model_big(modeljsonfile, dropout, lr):
    in_row, in_col, in_depth = 160, 320, 3
    model = Sequential()

    # normalize image values between -.5 : .5
    model.add(Lambda(lambda x: x / 255.0 - .5, input_shape=(in_row, in_col, in_depth)))

    ## horizon: 70, front of car: 25
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(in_row, in_col, in_depth)))

    # valid border mode should get rid of a couple each way, whereas same keeps
    # Use relu (non-linear activation function), not mentioned in Nvidia paper but a standard
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    # compile with normal adam optimizer (loss .001) and return
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    with open(modeljsonfile, 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
    return model


'''
Tested the comma.ai model on Nvidia's image size.
'''


def comma_model():
    row, col, depth = 66, 200, 3
    shape = (row, col, depth)

    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=shape, output_shape=shape))
    model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), padding='same'))
    model.add(ELU())
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(ELU())
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())

    # the fully connected layer accounts for huge % of parameters (50+)
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.summary()
    with open(model_dir + '/autopilot_comma.json', 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
    return model
