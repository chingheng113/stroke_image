import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.optimizers
import numpy as np
from data import data_util
from sklearn.metrics import classification_report, confusion_matrix
import os
import nibabel as nib
from results import result_util


def labelize(y_arr):
    y_label = []
    for y in y_arr:
        y_label = np.append(y_label, np.argmax(y))
    return y_label


def plot_training_acc(model_name):
    with open(model_name+'_trainHistory.pickle', 'rb') as f:
        history = pickle.load(f)
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.show()


def plot_training_loss(model_name):
    with open(model_name+'_trainHistory.pickle', 'rb') as f:
        history = pickle.load(f)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.show()


def load_testing_data():
    config = dict()
    config['which_machine'] = 'mri'
    config['n_classes'] = 2
    config["all_sequences"] = ['dwi', 'flair']
    config['n_channels'] = len(config["all_sequences"])
    config['image_shape'] = (20, 256, 256)

    read_test_file_path = os.path.join('..', 'data', 'mri', 'mri_data_testing.h5')

    id_data = data_util.open_data_file(read_test_file_path).root.id[:]
    X_data = data_util.open_data_file(read_test_file_path).root.data[:]
    # for i in range(3):
    #     result_util.save_array_to_nii(X_data[i], str(id_data[i]) + '_fs')
    y_data_o = data_util.open_data_file(read_test_file_path).root.label[:]
    y_data = keras.utils.to_categorical(y_data_o, num_classes=config['n_classes'])
    return X_data, y_data, y_data_o


def save_array_to_nii(X_data, fileName):
    new_image = nib.Nifti1Image(np.transpose(X_data), affine=np.eye(4))
    nib.save(new_image, fileName+'.nii')


if __name__ == '__main__':
    model_name = 'simple_VoxCNN'
    plot_training_acc(model_name)
    plot_training_loss(model_name)
    model = load_model(model_name+'.h5')
    X_data, y_data, y_data_o = load_testing_data()
    print(y_data_o[y_data_o == 0].shape)
    print(y_data_o[y_data_o == 1].shape)
    loss, acc = model.evaluate(X_data, y_data, verbose=0)
    predict_prob = model.predict(X_data)
    predict_labels = labelize(predict_prob)
    cm = confusion_matrix(y_data_o, predict_labels)
    print(cm)
    print(acc)