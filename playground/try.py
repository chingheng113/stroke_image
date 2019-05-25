import SimpleITK as sitk
import nibabel as nib
from results import result_util
import numpy as np
from matplotlib import pyplot as plt
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import scipy


heme_o = sitk.ReadImage('101_DWI.nii')
heme = sitk.GetArrayFromImage(heme_o)
print(heme.shape)
# rotate left 90
heme_RL90 = np.rot90(heme, axes=(1, 2))
# rotate right 90
heme_RR90 = np.rot90(heme, axes=(-1, -2))
# up side down
heme_up_down = flip(heme, axis=1)
# flip
heme_flipped = flip(heme, axis=2)
# crop
heme_crop = resize_image_with_crop_or_pad(heme, img_size=(20, 230, 230), mode='constant')
heme_crop = scipy.ndimage.zoom(heme_crop, (1, 256/230, 256/230), order=3)
print(heme_crop.shape)
result_util.save_array_to_nii(heme_crop, 'watch')


#
#
# oth = sitk.ReadImage('IXI012.nii')
# oth = sitk.GetArrayFromImage(oth)
# print(oth.shape)
#
#
# t1 = sitk.ReadImage('t1.nii')
# t1 = sitk.GetArrayFromImage(t1)
# print(t1.shape)