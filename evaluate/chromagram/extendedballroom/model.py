import numpy as np
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import sys, os

HOME_DIR = '/home/anupambiswas/Yeshwant'

sys.path.append(os.path.join(HOME_DIR, 'genre_classification', 'code', 'evaluate'))
from utils import load_data

feature, dataset = 'chromagram', 'extendedballroom'

def prepare_dataset(test_size):
    # load data
    data = load_data(feature, dataset)
    X, y = data['chroma'], data['label']
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    X = X.astype('float32')
    
    # create train/test split
    X_train, X_test, y_train, y_test =  train_test_split(X,
                                                         y,
                                                         test_size=test_size,
                                                        shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test    


def plot_history(history):
    with open('history.json', 'w') as fp:
        json.dump(history.history, fp, indent=4)

    fig, ax = plt.subplots(2, figsize=(10,8))

    # create accuracy subplot
    ax[0].plot(history.history['acc'], label='train accuracy')
    ax[0].plot(history.history['val_acc'], label='test accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='lower right')
    ax[0].set_title('Accuracy eval')

    # create error subplot
    ax[1].plot(history.history['loss'], label='train error')
    ax[1].plot(history.history['val_loss'], label='test error')
    ax[1].set_ylabel('Error')
    ax[1].set_xlabel('Epochs')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Error eval')

    plt.savefig('model1_' + feature + '.png')
    

def build_model(shape, output):
    model = models.Sequential()
    # layer 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=shape))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.1))
    # layer 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.1))
    # layer 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.1))
    # last layer
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(output, activation='softmax'))
    return model


def main():
    X_train, X_test, y_train, y_test = prepare_dataset(0.20)
    model = build_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]), 13)
    model.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, validation_split=0.2,
                epochs=100, batch_size=64, callbacks= [
                keras.callbacks.TensorBoard(log_dir='logs',
                                            histogram_freq=1,
                                            embeddings_freq=1)])

    plot_history(history)
    plot_model(model, show_shapes=True, to_file='model1.png')


if __name__ == '__main__':
    main()
