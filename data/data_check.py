import SimpleITK as sitk
import pandas as pd
import glob
import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
from data import data_util
current_path = os.path.dirname(__file__)

def check_image_shape():
    image_shape = (256, 256, 20)
    sequences = ['dwi', 'gre']
    for s in sequences:
        print('== ' + s)
        mri_path = os.path.join(current_path, 'mri')
        sq_path = os.path.join(mri_path, 'n4_' + s)
        read_path = os.path.join(sq_path, '*_' + s.upper() + '.nii')
        img_paths = glob.glob(read_path)
        for img_path in img_paths:
            sid = data_util.get_subject_id_from_path(img_path)
            img = sitk.ReadImage(img_path)
            img_data = sitk.GetArrayFromImage(img).T
            if img_data.shape != image_shape:
                print(sid)


def check_patient_number_cross_sequence():
    sequences = ['dwi', 'gre']
    id_list_sequences = []
    for s in sequences:
        mri_path = os.path.join(current_path, 'mri_all')
        sq_path = os.path.join(mri_path, 'n4_' + s)
        read_path = os.path.join(sq_path, '*_' + s.upper() + '.nii')
        img_paths = glob.glob(read_path)
        id_list = []
        for img_path in img_paths:
            sid = data_util.get_subject_id_from_path(img_path)
            id_list.append(sid)
        uniq_id_list = list(dict.fromkeys(id_list))
        id_list_sequences.append(uniq_id_list)
    print(list(set(id_list_sequences[0]) - set(id_list_sequences[1])))


if __name__ == '__main__':
    check_image_shape()
    check_patient_number_cross_sequence()

