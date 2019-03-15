import SimpleITK as sitk
from data import data_util
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import nibabel as nib

img_paths = data_util.get_img_paths('ct')
sitk_t1 = sitk.ReadImage(img_paths[0])
t1 = sitk.GetArrayFromImage(sitk_t1)
# Normalise the image to fit [-1, 1] range:
t1_norm_oo = normalise_one_one(t1)
# Add a feature dimension and normalise the image to fit [-1, 1] range
t1_norm_expand = np.expand_dims(normalise_one_one(t1), axis=-1)
# Randomly flip the image along axis 1
t1_flipped = flip(t1_norm_oo.copy(), axis=2)

print(t1.shape)
a = np.transpose(t1_flipped)
print(a.shape)
new_image = nib.Nifti1Image(a, affine=np.eye(4))
nib.save(new_image, '154a_ct.nii')
print('done')