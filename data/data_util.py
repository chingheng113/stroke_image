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
    replace_dic = {'Hemorrhage(Primary Hematoma)': 0,
                   'Hemorrhage(SDH)': 0,
                   'Hemorrhage(SDH), Other Diagnosis': 0,
                   'Ischemic Stroke': 1,
                   'TIA': 2,
                   'Other Diagnosis': 3
                   }
    labels.replace(replace_dic, inplace=True)
    id_label = pd.concat([ids, labels], axis=1)
    return id_label


def get_img_paths(mc):
    img_paths = ''
    if mc == 'mri':
        # to be continue...
        pass
    elif mc == 'ct':
        ct_img_path = os.path.join(current_path, 'ct')
        img_paths = glob.glob(os.path.join(ct_img_path, '*_ct.nii'))
    else:
        print('wrong parameter')
    return img_paths


def get_subject_id_from_path(img_path):
    file_name = img_path.split(os.sep)[-1]
    subject_id = file_name.split('_')[0]
    return subject_id


def get_subject_label(s_id, ids_labels):
    id_label = ids_labels[ids_labels.iloc[:, 0] == int(s_id)]
    label = id_label.iloc[:,1].values
    return label

def create_data_file(config, n_samples, file_path):
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


def write_ct_image_data_to_file(config, img_paths, data_storage, label_storage):
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


def write_data_to_file(config):
    file_path = os.path.join(current_path, config['which_machine'], config['which_machine'] + '_data.h5')
    img_paths = get_img_paths(config['which_machine'])
    n_samples = len(img_paths)
    hdf5_file, data_storage, label_storage = create_data_file(config, n_samples, file_path)
    if config['which_machine'] == 'ct':
        write_ct_image_data_to_file(config, img_paths, data_storage, label_storage)
        # write_ct_labels_to_file(config, img_paths)
    else:
        # MRI...to be continue...
        pass

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