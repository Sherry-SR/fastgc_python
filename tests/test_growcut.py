""" Tests for the growcut module """

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import nibabel as nib
from scipy.ndimage.morphology import binary_dilation, binary_erosion

from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__),'../modules')))

from growcut_fast import fastgc

def test_fastgrowcut():
    
    imgpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001.nii.gz"
    labelpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001_label_tri.nii.gz"
    savepath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/fastgc_python/results"

    img = nib.load(imgpath)
    imgdata = img.get_fdata()[420:500, 130:220, 223:230]
    label = nib.load(labelpath)
    labeldata = label.get_fdata()[420:500, 130:220, 223:230]
    seedsdata = np.zeros(labeldata.shape)

    nlabels = np.unique(labeldata)
    for i in nlabels:
        mask = labeldata == i
        mask = binary_erosion(mask)
        mask = binary_erosion(mask)
        seedsdata = seedsdata + mask * (i+1)

    distPre, labPre = fastgc(imgdata, seedsdata, newSeg = True, verbose = False)

    nib.save(nib.Nifti1Image(imgdata, img.affine), join(savepath, "original_img.nii.gz"))
    nib.save(nib.Nifti1Image(labeldata, img.affine), join(savepath, "original_label.nii.gz"))
    nib.save(nib.Nifti1Image(seedsdata, img.affine), join(savepath, "seedsdata.nii.gz"))
    nib.save(nib.Nifti1Image(labPre, img.affine), join(savepath, "fast_growcut_results.nii.gz"))

#test_fastgrowcut()