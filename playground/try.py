import SimpleITK as sitk
import nibabel as nib
from results import result_util
import numpy as np
from matplotlib import pyplot as plt
from dltk.io.augmentation import *
from dltk.io.preprocessing import *



heme_o = sitk.ReadImage('101_GRE.nii')
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
heme_crop = resize_image_with_crop_or_pad(heme, img_size=(20, 240, 240), mode='constant')
resample = sitk.ResampleImageFilter()
resample.SetSize((20, 256, 256))
resample.SetInterpolator(sitk.sitkLinear)
# resample.SetOutputSpacing(heme_o.GetSpacing())
resample.SetOutputDirection(heme_o.GetDirection())
resample.SetOutputOrigin(heme_o.GetOrigin())
resample.SetTransform(sitk.Transform())
# resample.SetDefaultPixelValue(heme_o.GetPixelIDValue())
a = sitk.GetImageFromArray(heme_crop)
resampled_sitk_img =resample.Execute(a)
heme_crop = sitk.GetArrayFromImage(resampled_sitk_img)
heme_crop = np.transpose(heme_crop)
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