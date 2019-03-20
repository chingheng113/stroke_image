import pandas as pd
import glob
import os
import shutil


if __name__ == '__main__':
    # create folder
    he_dir = os.path.join('..', os.path.join("data", "he"))
    if not os.path.exists(he_dir):
        os.mkdir(he_dir)
    is_dir = os.path.join('..', os.path.join("data", "is"))
    if not os.path.exists(is_dir):
        os.mkdir(is_dir)
    oth_dir = os.path.join('..', os.path.join("data", "oth"))
    if not os.path.exists(oth_dir):
        os.mkdir(oth_dir)
    tia_dir = os.path.join('..', os.path.join("data", "tia"))
    if not os.path.exists(tia_dir):
        os.mkdir(tia_dir)
    #
    df = pd.read_csv('HEME-ER details for Dr Fann.csv').dropna()
    dwi_pathes = glob.glob(r'C:\Users\linc9\Desktop\MRI_NIFTI\*_dwi.nii')
    for inx, dwi_path in enumerate(dwi_pathes):
        file_name = dwi_path.split(os.sep)[-1]
        subject_number = file_name.split('_')[0]
        label = df[df['MRI HEME #'].astype(int).astype(str) == subject_number]['Diagnosis Classification'].values[0]
        if 'Hemorrhage' in label:
            shutil.move(dwi_path, os.path.join(he_dir, file_name))
        elif 'Ischemic' in label:
            shutil.move(dwi_path, os.path.join(is_dir, file_name))
        elif 'Other Diagnosis' in label:
            shutil.move(dwi_path, os.path.join(oth_dir, file_name))
        elif 'TIA' in label:
            shutil.move(dwi_path, os.path.join(tia_dir, file_name))
        else:
            print(subject_number+' can not be classified')
    print('done')