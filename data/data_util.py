import pandas as pd
import glob
import os


def get_ids_labels(mca):
    df = pd.read_csv('HEME-ER details for Dr Fann.csv').dropna()
    # id
    if mca == 'mri':
        ids = df[['MRI HEME #']].astype(int)
    elif mca == 'ct':
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


if __name__ =='__main__':
    a = get_img_paths('ct')
    print('done')