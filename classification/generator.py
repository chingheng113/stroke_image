import sys, os
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
import copy
import pickle
from random import shuffle
import keras.optimizers
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import scipy
from results import result_util

def random_boolean():
    return np.random.choice([True, False])


# *************************************************************************
# make sure we are using pytables (conda install pytables), not tables !!!
# otherwise the Segmentation fault will drive you nuts...
# https://github.com/ellisdg/3DUnetCNN/issues/82
# https://github.com/ellisdg/3DUnetCNN/issues/58
# *************************************************************************


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


def augment_image(x_list, y_list, data_file, index, config):
    data = data_file.root.data[index]
    label_o = data_file.root.label[index]
    label = keras.utils.to_categorical(label_o, num_classes=config['n_classes'])
    # if random_boolean():
    # if True:
    #     # rotate left 90
    #     new_data = np.copy(img.get_data())
    #     img_RL90 = np.rot90(new_data, axes=(1, 2))
    #     result_util.save_array_to_nii(img_RL90[1,:,:,:], 'watchL90')
    #     x_list.append(img_RL90)
    #     y_list.append(label)
    # if True:
    #     # rotate right 90
    #     new_data = np.copy(img)
    #     img_RR90 = np.rot90(new_data, axes=(-1, -2))
    #     result_util.save_array_to_nii(img_RL90[1, :, :, :], 'watchR90')
    #     x_list.append(img_RR90)
    #     y_list.append(label)
    # if True:
    #     # up side down
    #     new_data = np.copy(img)
    #     img_up_down = flip(new_data, axis=1)
    #     result_util.save_array_to_nii(img_up_down[1, :, :, :], 'watchUp')
    #     x_list.append(img_up_down)
    #     y_list.append(label)
    if True:
        # flip
        sequnce_img = []
        for i in range(config['n_channels']):
            img = data[i, :, :, :]
            img_flipped = flip(img, axis=2)
            sequnce_img.append(img_flipped)
        flip_result = np.asarray(sequnce_img[:config['n_channels']])
        x_list.append(flip_result)
        y_list.append(label)
    if True:
        # crop
        sequnce_img = []
        for i in range(config['n_channels']):
            img = data[i, :, :, :]
            img_dim1 = config['image_shape'][1]
            img_dim2 = config['image_shape'][2]
            crop_dim1 = img_dim1 - 20
            crop_dim2 = img_dim2 - 20
            img_crop = resize_image_with_crop_or_pad(img, img_size=(config['image_shape'][0], crop_dim1, crop_dim2),
                                                     mode='constant')
            img_resize = scipy.ndimage.zoom(img_crop, (1, img_dim1/crop_dim1, img_dim2/crop_dim2), order=3)
            sequnce_img.append(img_resize)
        crop_result = np.asarray(sequnce_img[:config['n_channels']])
        x_list.append(crop_result)
        y_list.append(label)


def add_data(x_list, y_list, data_file, index, config):
    X_data = data_file.root.data[index]
    id = data_file.root.id[index]
    y_data_o = data_file.root.label[index]
    y_data = keras.utils.to_categorical(y_data_o, num_classes=config['n_classes'])
    x_list.append(X_data)
    y_list.append(y_data)
    augment_image(x_list, y_list, data_file, index, config)



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
            if len(x_list) == config['batch_size'] or len(x_list) > config['batch_size'] or (len(index_list) == 0 and len(x_list) > 0):
                # result_util.save_array_to_nii(x_list[1,index,:,:], 'watch_' + str(index))
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
    training_list, validation_list = get_validation_split(data_file, validation_size=0.2)
    train_generator = data_generator(data_file, training_list, config)
    validation_generator = data_generator(data_file, validation_list, config)
    num_training_steps = get_number_of_steps(len(training_list), config['batch_size'])
    num_validation_steps = get_number_of_steps(len(validation_list), config['batch_size'])
    return train_generator, validation_generator, num_training_steps , num_validation_steps
