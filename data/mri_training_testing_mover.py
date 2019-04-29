from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import os
import shutil
current_path = os.path.dirname(__file__)

if __name__ == '__main__':
    # ischmic (1) vs. non-ischemic (0)
    replace_dic = {'Hemorrhage(Primary Hematoma)': 0,
                   'Hemorrhage(SDH)': 0,
                   'Hemorrhage(SDH), Other Diagnosis': 0,
                   'Ischemic Stroke': 1,
                   'TIA': 0,
                   'Other Diagnosis': 0
                   }
    df = pd.read_csv(os.path.join(current_path, 'HEME-ER details for Dr Fann.csv')).dropna()
    ids = df[['MRI HEME #']].astype(int)
    labels = df[['Diagnosis Classification']]
    labels.replace(replace_dic, inplace=True)
    id_train, id_test, label_train, label_test = train_test_split(ids, labels, test_size=0.33)
    source_path = os.path.join(current_path, 'mri_all')
    # DWI ===
    argumentation = ['_R10', '_L10', '_flip', '_flip_L10', '_flip_R10']
    training_dwi_path = os.path.join(current_path, 'mri', 'training', 'n4_dwi')
    training_flair_path = os.path.join(current_path, 'mri', 'training', 'n4_flair')
    testing_dwi_path = os.path.join(current_path, 'mri', 'testing', 'n4_dwi')
    testing_flair_path = os.path.join(current_path, 'mri', 'testing', 'n4_flair')
    # deleting files
    shutil.rmtree(training_dwi_path)
    shutil.rmtree(training_flair_path)
    shutil.rmtree(testing_dwi_path)
    shutil.rmtree(testing_flair_path)
    # moving DWI data
    for index, row in id_train.iterrows():
        train_id = str(row['MRI HEME #'])
        a = os.path.join(source_path, 'n4_dwi', train_id+'_DWI.nii')
        shutil.copy(a, training_dwi_path)
        b = 0




    # FLAIR
    testing_dwi_path = os.path.join(current_path, 'mri', 'testing', 'n4_dwi')
    testing_flair_path = os.path.join(current_path, 'mri', 'testing', 'n4_flair')

