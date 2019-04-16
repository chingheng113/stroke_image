import SimpleITK as sitk
import pandas as pd
import glob
import os
import numpy as np
import tables

current_path = os.path.dirname(__file__)


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


def get_mr_sequence_paths(config):
    mri_path = os.path.join(current_path, 'mri')
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
        raise Exception('the sample size if each sequence is not consist')

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


def write_ct_image_label_to_file(config, img_paths, data_storage, label_storage):
    ct_img_list = []
    ids_labels = get_ids_labels(config['which_machine'])
    for img_path in img_paths:
        img = sitk.ReadImage(img_path)
        sid = get_subject_id_from_path(img_path)
        true_label = get_subject_label(sid, ids_labels)
        ct_img_list.append(sitk.GetArrayFromImage(img).T)
        data_storage.append(np.asarray(ct_img_list[:config['n_channels']])[np.newaxis])
        label_storage.append(true_label)
    return data_storage, label_storage


def write_mr_image_label_to_file(config, sequence_paths, data_storage, label_storage):
    # At this monent, we sure that sample size of each sequence are consist
    # But I still want to go one by one to make sure the order of samples and label is correct
    # it is very inefficient way, but... be careful
    ids_labels = get_ids_labels(config['which_machine'])
    mri_path = os.path.join(current_path, 'mri')
    img_paths_for_loop = glob.glob(os.path.join(sequence_paths[0], '*.nii'))
    for img_path_for_loop in img_paths_for_loop:
        sid = get_subject_id_from_path(img_path_for_loop)
        mr_img_list = []
        for s in config['all_sequences']:
            sq_path = os.path.join(mri_path, 'n4_'+s)
            read_path = os.path.join(sq_path, sid+'_'+s.upper()+'.nii')
            img = sitk.ReadImage(read_path)
            img_data = sitk.GetArrayFromImage(img).T
            mr_img_list.append(img_data)
        data_storage.append(np.asarray(mr_img_list[:config['n_channels']])[np.newaxis])
        true_label = get_subject_label(sid, ids_labels)
        label_storage.append(true_label)
    return data_storage, label_storage


def write_data_to_file(config):
    file_path = os.path.join(current_path, config['which_machine'], config['which_machine'] + '_data.h5')
    if config['which_machine'] == 'ct':
        img_paths = get_ct_img_paths()
        n_samples = len(img_paths)
        hdf5_file, data_storage, label_storage = create_ct_data_file(config, n_samples, file_path)
        write_ct_image_label_to_file(config, img_paths, data_storage, label_storage)
    else:
        # MRI
        sequence_paths = get_mr_sequence_paths(config)
        hdf5_file, data_storage, label_storage = create_mr_data_file(config, sequence_paths, file_path)
        write_mr_image_label_to_file(config, sequence_paths, data_storage, label_storage)
        a = 1
    hdf5_file.close()
    return file_path


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)


if __name__ =='__main__':
    config = dict()
    config['image_shape'] = (64, 64, 64)
    config['which_machine'] = 'ct'
    config['n_channels'] = 1
    write_data_to_file(config)
    print('done')