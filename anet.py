import keras.optimizers
from tensorflow import keras
from tensorflow.keras import layers, models
from config import HexConfig as config
import tensorflow as tf


def init_ANN_CNN():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), input_shape=config.CNN_INPUT_DIM,
                            data_format="channels_last"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    # model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    # model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1)))
    model.add(layers.Flatten())
    """for i in range(len(DIMENSIONS)):
        model.add(layers.Dense(DIMENSIONS[i], activation="relu"))"""
    model.add(layers.Dense(config.OUTPUT_DIM, activation="softmax"))
    model.summary()
    if config.OPT == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=config.LR)
    elif config.OPT == "SGD":
        opt = tf.keras.optimizers.SGD(learning_rate=config.LR)
    elif config.OPT == "RMSprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=config.LR)
    elif config.OPT == "Adagrad":
        opt = tf.keras.optimizers.Adagrad(learning_rate=config.LR)
    else:
        opt = None

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    print(type(model))
    return model


def init_ANN():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=config.INPUT_DIM, batch_size=config.BATCH_SIZE))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(config.OUTPUT_DIM, activation="softmax"))
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    print(type(model))
    return model
