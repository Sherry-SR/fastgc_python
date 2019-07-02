""" Tests for the growcut module """

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import nibabel as nib
import time
from scipy.ndimage.morphology import binary_dilation, binary_erosion

from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__),'../modules')))

from growcut_cpu import fastgc

def test_fastgrowcut():
    """
    # Teeth
    imgpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001.nii.gz"
    labelpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001_label_tri.nii.gz"
    savepath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/fastgc_python/results"

    img = nib.load(imgpath)
    imgdata = np.squeeze(img.get_fdata()[120:580, 10:350, 225:226])
    label = nib.load(labelpath)
    labeldata = np.squeeze(label.get_fdata()[120:580, 10:350, 225:226])
    seedsdata = np.zeros(labeldata.shape)

    """

    # Pelvis
    imgpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/pelvis03_ct.nii.gz"
    labelpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/pelvis03-ct_label.nii.gz"
    savepath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/fastgc_python/results"

    img = nib.load(imgpath)
    imgdata = np.squeeze(img.get_fdata()[80:460, 210:420, 149:150])
    label = nib.load(labelpath)
    labeldata = np.squeeze(label.get_fdata()[80:460, 210:420, 149:150])
    seedsdata = np.zeros(labeldata.shape)

    nlabels = np.unique(labeldata)
    for i in nlabels:
        mask = labeldata == i
        mask = binary_erosion(mask, structure=np.ones((3,3)))
        mask = binary_erosion(mask, structure=np.ones((3,3)))
        mask = binary_erosion(mask, structure=np.ones((3,3)))
        mask = binary_erosion(mask, structure=np.ones((3,3)))
        mask = binary_erosion(mask, structure=np.ones((3,3)))

        seedsdata = seedsdata + mask * (i+1)

    start = time.time()
    distPre, labPre = fastgc(imgdata, seedsdata, newSeg = True, verbose = True)
    end = time.time()
    print("time used:", end - start, "seconds")
    """
    # Teeth
    nib.save(nib.Nifti1Image(imgdata, img.affine), join(savepath, "original_img_teeth.nii.gz"))
    nib.save(nib.Nifti1Image(labeldata, img.affine), join(savepath, "original_label_teeth.nii.gz"))
    nib.save(nib.Nifti1Image(seedsdata, img.affine), join(savepath, "seedsdata_teeth.nii.gz"))
    nib.save(nib.Nifti1Image(labPre, img.affine), join(savepath, "fast_growcut_results_teeth.nii.gz"))
    
"""
    # Pelvis
    nib.save(nib.Nifti1Image(imgdata, img.affine), join(savepath, "original_img_pelvis.nii.gz"))
    nib.save(nib.Nifti1Image(labeldata, img.affine), join(savepath, "original_label_pelvis.nii.gz"))
    nib.save(nib.Nifti1Image(seedsdata, img.affine), join(savepath, "seedsdata_pelvis.nii.gz"))
    nib.save(nib.Nifti1Image(labPre, img.affine), join(savepath, "fast_growcut_results_pelvis.nii.gz"))

test_fastgrowcut()