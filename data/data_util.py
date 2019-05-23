import SimpleITK as sitk
import pandas as pd
import glob
import os, sys
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
import numpy as np
import tables
import pickle
from dltk.io.preprocessing import normalise_zero_one
from results import result_util

current_path = os.path.dirname(__file__)


def random_boolean():
    return np.random.choice([True, False])


def get_ids_labels(mc):
    df = pd.read_csv(os.path.join(current_path, 'HEME-ER details for Dr Fann.csv')).dropna()
    # id
    if mc == 'mri':
        ids = df[['MRI HEME #']].astype(int)
    elif mc == 'ct':
        ids = df[['CT HEME #']].astype(int)
    else:
        ids = df[['MRI HEME #', 'CT HEME #']].astype(int)
    # label
    labels = df[['Diagnosis Classification']]
    # ischmic (1) vs. non-ischemic (0)
    replace_dic = {'Hemorrhage(Primary Hematoma)': 0,
                   'Hemorrhage(SDH)': 0,
                   'Hemorrhage(SDH), Other Diagnosis': 0,
                   'Ischemic Stroke': 1,
                   'TIA': 0,
                   'Other Diagnosis': 0
                   }
    labels.replace(replace_dic, inplace=True)
    id_label = pd.concat([ids, labels], axis=1)
    return id_label


def get_ct_img_paths():
    ct_img_path = os.path.join(current_path, 'ct')
    img_paths = glob.glob(os.path.join(ct_img_path, '*_ct.nii'))
    return img_paths


def get_mr_sequence_paths(config, training_test):
    mri_path = os.path.join(current_path, 'mri', training_test)
    sequence_paths = []
    for s in config['all_sequences']:
        sequence_path = os.path.join(mri_path, 'n4_'+s)
        sequence_paths.append(sequence_path)
    return sequence_paths


def get_subject_id_from_path(img_path):
    file_name = img_path.split(os.sep)[-1]
    subject_id = file_name.split('_')[0]
    return subject_id


def get_subject_label(s_id, ids_labels):
    id_label = ids_labels[ids_labels.iloc[:, 0] == int(s_id)]
    label = id_label.iloc[:,1].values
    return label


def create_ct_data_file(config, n_samples, file_path):
    hdf5_file = tables.open_file(file_path, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, config['n_channels']] + list(config['image_shape']))
    try:
        data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                               filters=filters, expectedrows=n_samples)
        label_storage = hdf5_file.create_earray(hdf5_file.root, 'label', tables.Float32Atom(), shape=(0,),
                                               filters=filters, expectedrows=n_samples)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(file_path)
        raise e
    return hdf5_file, data_storage, label_storage


def create_mr_data_file(config, sequence_paths, file_path):
    sample_size = []
    for sequence_path in sequence_paths:
        img_paths = glob.glob(os.path.join(sequence_path, '*.nii'))
        sample_size.append(len(img_paths))

    if np.unique(sample_size).shape[0] <= 1:
        # check all sequence have same sample size
        n_samples = sample_size[0]
    else:
        raise Exception('the sample size of each sequence is not consist')

    hdf5_file = tables.open_file(file_path, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, config['n_channels']] + list(config['image_shape']))
    try:
        data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                               filters=filters, expectedrows=n_samples)
        label_storage = hdf5_file.create_earray(hdf5_file.root, 'label', tables.Float32Atom(), shape=(0,),
                                                filters=filters, expectedrows=n_samples)
        id_storage = hdf5_file.create_earray(hdf5_file.root, 'id', tables.StringAtom(itemsize=10), shape=(0,),
                                             filters=filters, expectedrows=n_samples)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(file_path)
        raise e
    return hdf5_file, data_storage, label_storage, id_storage


def write_ct_image_label_to_file(config, img_paths, data_storage, label_storage):
    ct_img_list = []
    ids_labels = get_ids_labels(config['which_machine'])
    for img_path in img_paths:
        img = sitk.ReadImage(img_path)
        img_arr = sitk.GetArrayFromImage(img)
        ct_img_list.append(img_arr)
        sid = get_subject_id_from_path(img_path)
        true_label = get_subject_label(sid, ids_labels)
        data_storage.append(np.asarray(ct_img_list[:config['n_channels']])[np.newaxis])
        label_storage.append(true_label)
    return data_storage, label_storage


def write_mr_image_label_to_file(config, training_or_testing, sequence_paths, data_storage, label_storage, id_storage):
    # At this monent, we sure that sample size of each sequence are consist
    # But I still want to go one by one to make sure the order of samples and label is correct
    # it is very inefficient way, but... be careful
    ids_labels = get_ids_labels(config['which_machine'])
    mri_path = os.path.join(current_path, 'mri', training_or_testing)
    temp_sequence = config['all_sequences'][0].upper()
    img_paths_for_loop = glob.glob(os.path.join(sequence_paths[0], '*_'+temp_sequence+'.nii'))
    id_list = []
    for img_path_for_loop in img_paths_for_loop:
        sid = get_subject_id_from_path(img_path_for_loop)
        mr_img_list = []
        for s in config['all_sequences']:
            sq_path = os.path.join(mri_path, 'n4_'+s)
            read_path = os.path.join(sq_path, sid+'_'+s.upper()+'.nii')
            img = sitk.ReadImage(read_path)
            img_data = sitk.GetArrayFromImage(img)
            img_data = normalise_zero_one(img_data)
            # result_util.save_array_to_nii(img_data, sid+'_'+s+'_f')
            mr_img_list.append(img_data)
        data_storage.append(np.asarray(mr_img_list[:config['n_channels']])[np.newaxis])
        true_label = get_subject_label(sid, ids_labels)
        label_storage.append(true_label)
        id_storage.append([sid])
        id_list.append(sid)
    if training_or_testing == 'training':
        data_storage, label_storage, id_storage = add_augmentation(config, id_list, data_storage, label_storage, id_storage)
    return data_storage, label_storage, id_storage


def add_augmentation(config, id_list, data_storage, label_storage, id_storage):
    mri_path = os.path.join(current_path, 'mri', 'training')
    ids_labels = get_ids_labels(config['which_machine'])
    for id in id_list:
        for augment in config['augments']:
            if random_boolean():
            # if True:
                mr_img_list = []
                for s in config['all_sequences']:
                    sq_path = os.path.join(mri_path, 'n4_' + s)
                    read_path = os.path.join(sq_path, id + '_' + s.upper() + '_'+augment+'.nii')
                    img = sitk.ReadImage(read_path)
                    img_data = sitk.GetArrayFromImage(img)
                    img_data = normalise_zero_one(img_data)
                    mr_img_list.append(img_data)
                data_storage.append(np.asarray(mr_img_list[:config['n_channels']])[np.newaxis])
                true_label = get_subject_label(id, ids_labels)
                label_storage.append(true_label)
                id_storage.append([id])
    return data_storage, label_storage, id_storage


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_training_data_storage(data_storage, config):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    train_mean = np.asarray(means).mean(axis=0)
    train_std = np.asarray(stds).mean(axis=0)
    config['train_mean'] = train_mean
    config['train_std'] = train_std
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], train_mean, train_std)
    return data_storage, config


def normalize_testing_data_storage(data_storage, config):
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], config['train_mean'], config['train_std'])
    return data_storage


def write_data_to_file(config, training_or_testing):
    if training_or_testing != 'training' and training_or_testing != 'testing':
        raise Exception('Must be training or testing')
    file_path = os.path.join(current_path, config['which_machine'], config['which_machine'] + '_data_'+training_or_testing+'.h5')
    if config['which_machine'] == 'ct':
        img_paths = get_ct_img_paths()
        n_samples = len(img_paths)
        hdf5_file, data_storage, label_storage = create_ct_data_file(config, n_samples, file_path)
        write_ct_image_label_to_file(config, img_paths, data_storage, label_storage)
    else:
        # MRI
        sequence_paths = get_mr_sequence_paths(config, training_or_testing)
        hdf5_file, data_storage, label_storage, id_storage = create_mr_data_file(config, sequence_paths, file_path)
        write_mr_image_label_to_file(config, training_or_testing, sequence_paths, data_storage, label_storage, id_storage)
    if training_or_testing == 'training':
        normalize_training_data_storage(data_storage, config)
    else:
        normalize_testing_data_storage(data_storage, config)
    hdf5_file.close()
    return file_path


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)


def save_history(model_name, history):
    save_path = os.path.join('..', 'results', model_name+'_trainHistory.pickle')
    with open(save_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def save_model(model_name, model):
    save_path = os.path.join('..', 'results', model_name + '.h5')
    model.save(save_path)


if __name__ =='__main__':
    config = dict()
    config['image_shape'] = (64, 64, 64)
    config['which_machine'] = 'ct'
    config['n_channels'] = 1
    write_data_to_file(config)
    print('done')