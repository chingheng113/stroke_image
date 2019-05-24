import SimpleITK as sitk

heme = sitk.ReadImage('101_GRE.nii')
heme = sitk.GetArrayFromImage(heme)
print(heme.shape)

oth = sitk.ReadImage('IXI012.nii')
oth = sitk.GetArrayFromImage(oth)
print(oth.shape)