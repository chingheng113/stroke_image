import sys, os
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
import pickle
import numpy as np
from random import shuffle
import keras.optimizers


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def split_list(input_list, validation_size=0.3, shuffle_list=True):
    print('in split_list')
    split = 1. - validation_size
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def get_validation_split(data_file, validation_size):
    print("Creating validation split...")
    nb_samples = data_file.root.data.shape[0]
    sample_list = list(range(nb_samples))
    training_list, validation_list = split_list(sample_list, validation_size, True)
    # pickle_dump(training_list, 'training_ids.pkl')
    # pickle_dump(validation_list, 'testing_ids.pkl')
    return training_list, validation_list


def add_data(x_list, y_list, data_file, index, config):
    print('in add_data')
    print('before data_file')
    X_data = data_file.root.data[index]
    y_data_o = data_file.root.label[index]
    print('before keras')
    y_data = keras.utils.to_categorical(y_data_o, num_classes=config['n_classes'])
    print('after keras')
    x_list.append(X_data)
    y_list.append(y_data)


def convert_data(x_list, y_list):
    print('in covert_data')
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    return x, y


def data_generator(data_file, index_list, config):
    print('in data_generator')
    while True:
        x_list = list()
        y_list = list()
        shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, config)
            yield convert_data(x_list, y_list)
            x_list = list()
            y_list = list()


def get_training_and_validation_generators(data_file, config):
    training_list, validation_list = get_validation_split(data_file, validation_size=0.3)
    train_generator = data_generator(data_file, training_list, config)
    validation_generator = data_generator(data_file, validation_list, config)
    n_train_steps = 3
    n_validation_steps = 3
    return train_generator, validation_generator, n_train_steps, n_validation_steps
