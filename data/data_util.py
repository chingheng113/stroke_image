import SimpleITK as sitk
import pandas as pd
import glob
import os
import numpy as np


def get_ids_labels(mc):
    df = pd.read_csv('HEME-ER details for Dr Fann.csv').dropna()
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
    current_path = os.path.dirname(__file__)
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


def get_all_imgs(mc):
    read_pathes = get_img_paths(mc)
    a = []
    for read_path in read_pathes:
        sitk_t1 = sitk.ReadImage(read_path)
        t1 = sitk.GetArrayFromImage(sitk_t1).T.reshape(-1, 1, 64, 64, 64)
        a = np.stack(t1, axis=0)
    print(a)

if __name__ =='__main__':
    get_all_imgs('ct')
    print('done')