import sys, os
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
import copy
import pickle
import numpy as np
from random import shuffle
import keras.optimizers


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def split_list(input_list, validation_size=0.3, shuffle_list=True):
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
    X_data = data_file.root.data[index]
    y_data_o = data_file.root.label[index]
    y_data = keras.utils.to_categorical(y_data_o, num_classes=config['n_classes'])
    x_list.append(X_data)
    y_list.append(y_data)


def convert_data(x_list, y_list):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    return x, y


def data_generator(data_file, index_list, config):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        index_list = copy.copy(orig_index_list)
        shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, config)
            if len(x_list) == config['batch_size'] or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list)
                x_list = list()
                y_list = list()


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_training_and_validation_generators(data_file, config):
    training_list, validation_list = get_validation_split(data_file, validation_size=0.3)
    train_generator = data_generator(data_file, training_list, config)
    validation_generator = data_generator(data_file, validation_list, config)
    num_training_steps = get_number_of_steps(len(training_list), config['batch_size'])
    num_validation_steps = get_number_of_steps(len(validation_list), config['batch_size'])
    return train_generator, validation_generator, num_training_steps , num_validation_steps
