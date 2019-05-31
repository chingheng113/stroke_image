from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import sys, os
import shutil
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
from data import skip_ids


# ischmic (1) vs. non-ischemic (0)
ischmic_non_dic = {'Hemorrhage(Primary Hematoma)': 0,
                   'Hemorrhage(SDH)': 0,
                   'Hemorrhage(SDH), Other Diagnosis': 0,
                   'Ischemic Stroke': 1,
                   'TIA': 0,
                   'Other Diagnosis': 0
                   }


if __name__ == '__main__':
    current_path = os.path.dirname(__file__)
    file_path = os.path.join('..', 'data')
    source_path = os.path.join(file_path, 'mri_all')
    df = pd.read_csv(os.path.join(file_path, 'HEME-ER details for Dr Fann.csv')).dropna()
    # Remove skip ids ==
    skip_id_list = skip_ids.dwi_less_slice + skip_ids.dwi_noise + skip_ids.no_gre + skip_ids.gre_noise+ skip_ids.flair_noise + skip_ids.no_flair
    skip_id_list_uniq = list(dict.fromkeys(skip_id_list))
    df = df[~df['MRI HEME #'].isin(skip_id_list_uniq)]

    ids = df[['MRI HEME #']].astype(int)
    labels = df[['Diagnosis Classification']]
    labels.replace(ischmic_non_dic, inplace=True)
    print()
    isc_ids = ids.loc[labels[labels['Diagnosis Classification'] == 1].index]
    nisc_ids = ids.loc[labels[labels['Diagnosis Classification'] == 0].index]


    sequence = ['dwi', 'gre', 'flair']
    for s in sequence:
        for index, row in isc_ids.iterrows():
            id = str(row['MRI HEME #'])
            read_path = os.path.join(file_path, 'mri_all', 'n4_'+s, str(id)+'_'+s.upper()+'.nii')
            target_path = os.path.join(current_path, s, 'ischemic')
            shutil.copy(read_path, target_path)
        for index, row in nisc_ids.iterrows():
            id = str(row['MRI HEME #'])
            read_path = os.path.join(file_path, 'mri_all', 'n4_' + s, str(id) + '_' + s.upper() + '.nii')
            target_path = os.path.join(current_path, s, 'non-ischemic')
            shutil.copy(read_path, target_path)

    print('Moving files done')

