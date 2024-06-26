from keras import layers
from keras.models import Sequential
from keras import callbacks

import tensorflow as tf

tf.get_logger().setLevel('INFO')

from typing import List

import numpy as np
import pathlib
import os

import matplotlib.pyplot as plt

data_dir = pathlib.Path("pokemons/")
test_dir = pathlib.Path("test_data/")

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.is_dir() and item.name != ".ipynb_checkpoints"]))

class TrainingCallback(callbacks.Callback):

    loss: List[float] = []
    accuracy: List[float] = []
    val_loss: List[float] = []
    val_accuracy: List[float] = []


    def __init__(self):
        super().__init__()
        plt.ion()
        plt.show()

        plt.figure(1)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(self.val_loss, 'b-', label='Validation Loss')
        plt.plot(self.loss, 'g-', label='Loss')
        plt.legend()

        plt.figure(2)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(self.val_accuracy, 'r-', label='Validation Accuracy')
        plt.plot(self.accuracy, 'm-', label='Accuracy')

        plt.legend()


    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs['loss'])
        self.accuracy.append(logs['accuracy'])
        self.val_loss.append(logs['val_loss'])
        self.val_accuracy.append(logs['val_accuracy'])

        self.plot_figure1(epoch)
        self.plot_figure2(epoch)


    def plot_figure1(self, epoch: int):
        plt.figure(1)
        plt.xlim(0, 35) 
        plt.ylim(0, 5)
        
        plt.plot(self.val_loss, 'b-', label='Validation Loss')
        plt.plot(self.loss, 'g-', label='Loss')

        plt.draw()
        plt.pause(0.1)
        plt.savefig(f'epochs/epoch_loss{epoch}.png')

    def plot_figure2(self, epoch: int):
        plt.figure(2)
        plt.xlim(0, 35)
        plt.ylim(0, 1)

        plt.plot(self.val_accuracy, 'r-', label='Validation Accuracy')
        plt.plot(self.accuracy, 'm-', label='Accuracy')

        plt.draw()
        plt.pause(0.1)
        plt.savefig(f'epochs/epoch_accuracy{epoch}.png')

    def on_train_end(self, logs=None):
        plt.ioff()
        plt.show() # blocking show

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [224, 224])

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

data_augmentation = Sequential([
    layers.RandomZoom(0.3, fill_mode="nearest"),
    layers.RandomRotation(0.1),
    layers.RandomFlip("vertical"),
])

def create_model(num_classes: int):
    model = Sequential([
        layers.Rescaling(1./255, offset=-1),
        data_augmentation,

        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Dropout(0.1),
        
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])


    return model