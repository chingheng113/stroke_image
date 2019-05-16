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
testing_dwi_path = os.path.join(current_path, 'mri', 'testing', 'n4_dwi')
testing_flair_path = os.path.join(current_path, 'mri', 'testing', 'n4_flair')


def clean_folder():
    shutil.rmtree(training_dwi_path)
    os.mkdir(training_dwi_path)
    shutil.rmtree(training_flair_path)
    os.mkdir(training_flair_path)
    shutil.rmtree(testing_dwi_path)
    os.mkdir(testing_dwi_path)
    shutil.rmtree(testing_flair_path)
    os.mkdir(testing_flair_path)


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
    # deleting files
    clean_folder()
    # training data
    for index, row in id_train.iterrows():
        train_id = str(row['MRI HEME #'])
        if train_id not in skip_ids.dwi_less_slice+skip_ids.flair_noise+skip_ids.no_flair:
            shutil.copy(os.path.join(source_path, 'n4_dwi', train_id+'_DWI.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_L10', train_id + '_DWI_RL10.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_L90', train_id + '_DWI_RL90.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_R10', train_id + '_DWI_RR10.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_R90', train_id + '_DWI_RR90.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_flip', train_id + '_DWI_F.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_flip_L10', train_id + '_DWI_F_RL10.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_flip_R10', train_id + '_DWI_F_RR10.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_crop', train_id + '_DWI_c.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_crop_Flip', train_id + '_DWI_c_F.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_crop_Flip_L90', train_id + '_DWI_c_F_RL90.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_crop_Flip_R90', train_id + '_DWI_c_F_RR90.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_crop_L90', train_id + '_DWI_c_RL90.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_dwi_crop_R90', train_id + '_DWI_c_RR90.nii'), training_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_flair', train_id + '_FLAIR.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_L10', train_id + '_FLAIR_RL10.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_L90', train_id + '_FLAIR_RL90.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_R10', train_id + '_FLAIR_RR10.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_R90', train_id + '_FLAIR_RR90.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_flip', train_id + '_FLAIR_F.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_flip_L10', train_id + '_FLAIR_F_RL10.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_flip_R10', train_id + '_FLAIR_F_RR10.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_crop', train_id + '_FLAIR_c.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_crop_Flip', train_id + '_FLAIR_c_F.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_crop_Flip_L90', train_id + '_FLAIR_c_F_RL90.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_crop_Flip_R90', train_id + '_FLAIR_c_F_RR90.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_crop_L90', train_id + '_FLAIR_c_RL90.nii'), training_flair_path)
            shutil.copy(os.path.join(source_path, 'n4_flair_crop_R90', train_id + '_FLAIR_c_RR90.nii'), training_flair_path)

    # testinf data
    for index, row in id_test.iterrows():
        test_id = str(row['MRI HEME #'])
        if test_id not in skip_ids.dwi_less_slice+skip_ids.flair_noise+skip_ids.no_flair:
            shutil.copy(os.path.join(source_path, 'n4_dwi', test_id + '_DWI.nii'), testing_dwi_path)
            shutil.copy(os.path.join(source_path, 'n4_flair', test_id + '_FLAIR.nii'), testing_flair_path)
    print('Moving files done')

