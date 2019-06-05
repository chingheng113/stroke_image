from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import sys, os
import shutil
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
from data import skip_ids

current_path = os.path.dirname(__file__)
source_path = os.path.join(current_path, 'mri_all')
training_dwi_path = os.path.join(current_path, 'mri', 'training', 'n4_dwi')
training_flair_path = os.path.join(current_path, 'mri', 'training', 'n4_flair')
training_gre_path = os.path.join(current_path, 'mri', 'training', 'n4_gre')

testing_dwi_path = os.path.join(current_path, 'mri', 'testing', 'n4_dwi')
testing_flair_path = os.path.join(current_path, 'mri', 'testing', 'n4_flair')
testing_gre_path = os.path.join(current_path, 'mri', 'testing', 'n4_gre')


def clean_folder():
    shutil.rmtree(training_dwi_path)
    os.mkdir(training_dwi_path)
    # shutil.rmtree(training_flair_path)
    # os.mkdir(training_flair_path)
    shutil.rmtree(training_gre_path)
    os.mkdir(training_gre_path)
    shutil.rmtree(testing_dwi_path)
    os.mkdir(testing_dwi_path)
    # shutil.rmtree(testing_flair_path)
    # os.mkdir(testing_flair_path)
    shutil.rmtree(testing_gre_path)
    os.mkdir(testing_gre_path)


# ischmic (1) vs. non-ischemic (0)
ischmic_non_dic = {'Hemorrhage(Primary Hematoma)': 0,
                   'Hemorrhage(SDH)': 0,
                   'Hemorrhage(SDH), Other Diagnosis': 0,
                   'Ischemic Stroke': 1,
                   'TIA': 0,
                   'Other Diagnosis': 0
                   }
# ischmic_non_dic = {'Hemorrhage(Primary Hematoma)': 0,
#                    'Hemorrhage(SDH)': 0,
#                    'Hemorrhage(SDH), Other Diagnosis': 0,
#                    'TIA': 1
#                    }

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(current_path, 'HEME-ER details for Dr Fann.csv')).dropna()
    # Remove skip ids ==
    # skip_id_list = skip_ids.dwi_less_slice + skip_ids.flair_noise + skip_ids.no_flair +skip_ids.dwi_noise
    skip_id_list = skip_ids.dwi_less_slice + skip_ids.dwi_noise + skip_ids.no_gre + skip_ids.gre_noise
    skip_id_list_uniq = list(dict.fromkeys(skip_id_list))
    df = df[~df['MRI HEME #'].isin(skip_id_list_uniq)]
    print(df[df['Diagnosis Classification'] == 'Ischemic Stroke'].shape)
    print(df[df['Diagnosis Classification'] == 'TIA'].shape)
    print(df[df['Diagnosis Classification'] == 'Other Diagnosis'].shape)
    # only get Ischemic and TIA
    # df_less = df[df['Diagnosis Classification'].isin(['Hemorrhage(Primary Hematoma)', 'Hemorrhage(SDH)', 'Hemorrhage(SDH), Other Diagnosis'])]
    # df_less = df[df['Diagnosis Classification'] == 'TIA']
    # df_less = df[df['Diagnosis Classification'].isin(['TIA', 'Hemorrhage(Primary Hematoma)', 'Hemorrhage(SDH)', 'Hemorrhage(SDH), Other Diagnosis'])]
    # df_more = df[df['Diagnosis Classification'] == 'Ischemic Stroke']
    # df_more_down_sample = df_more.sample(n=df_less.shape[0],  random_state=123)
    # df = pd.concat([df_less, df_more_down_sample], axis=0)

    # only one class
    # df = df[df['Diagnosis Classification'] == 'Ischemic Stroke']

    ids = df[['MRI HEME #']].astype(int)
    labels = df[['Diagnosis Classification']]
    labels.replace(ischmic_non_dic, inplace=True)

    id_train, id_test, label_train, label_test = train_test_split(ids, labels, test_size=0.2)
    # deleting files
    clean_folder()
    # training data
    for index, row in id_train.iterrows():
        train_id = str(row['MRI HEME #'])
        shutil.copy(os.path.join(source_path, 'n4_dwi_mni', train_id+'_DWI.nii'), training_dwi_path)
        # shutil.copy(os.path.join(source_path, 'n4_flair', train_id + '_FLAIR.nii'), training_flair_path)
        # shutil.copy(os.path.join(source_path, 'n4_gre', train_id + '_GRE.nii'), training_gre_path)
    # testinf data
    for index, row in id_test.iterrows():
        test_id = str(row['MRI HEME #'])
        shutil.copy(os.path.join(source_path, 'n4_dwi_mni', test_id + '_DWI.nii'), testing_dwi_path)
        # shutil.copy(os.path.join(source_path, 'n4_flair', test_id + '_FLAIR.nii'), testing_flair_path)
        # shutil.copy(os.path.join(source_path, 'n4_gre', test_id + '_GRE.nii'), testing_gre_path)
    print('Moving files done')

