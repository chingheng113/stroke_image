import SimpleITK as sitk
import pandas as pd
import glob
import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
from data import data_util
current_path = os.path.dirname(__file__)

if __name__ == '__main__':
    image_shape = (256, 256, 20)
    sequences = ['dwi', 'flair']
    for s in sequences:
        print('== '+s)
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

