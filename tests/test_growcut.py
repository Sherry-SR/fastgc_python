""" Tests for the growcut module """

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__),'..')))

from growcut import growcut

def sample_data(shape):
    """ Forms a circle to segment """
    cx, cy = shape[1] // 2, shape[0] // 2)
    r = 5

    y, x = np.ogrid[0:shape[0], 0:shape[1]]
    mask = (np.power((y - cy), 2) + np.power((x - cx), 2)) < np.power(r, 2)

    image = np.zeros(mask.shape)
    image[mask] = 1.0

    label = np.zeros(mask.shape)
    strength = np.zeros(mask.shape)
    
    ind = r // 4
    label[0:ind, 0:ind] = -1
    strength[0:ind, 0:ind] = 1.0

    label[cy - ind:cy + ind, cx - ind:cx +ind] = 1
    strength[cy - ind:cy + ind, cx - ind:cx + ind] = 1.0

    return image, mask, label, strength


#@pytest.mark.parametrize(("shape"), [(15, 15), (20, 20), (40, 40)])
def test_growcut(shape):
    """ Test correct segmentations using growcut """

    image, mask, label, strength = sample_data(shape)

    segmentation = growcut.growcut(
        image,
        np.dstack((label, strength)),
        window_size=3)

    assert np.allclose(mask, segmentation == 1), "Segmentation did not converge"

#@pytest.mark.parametrize(("shape"), [(10, 10), (20, 20), (40, 40)])
#def test_growcut_cython_equality(shape):
#    """ Test correct segmentations using growcut """
#
#    image, mask, label, strength = sample_data(shape)
#
#    segmentation_slow = growcut.growcut(
#        image,
#        np.dstack((label, strength)),
#        window_size=3)
#
#    segmentation_fast = growcut_cy.growcut(
#        np.array([image, image, image]),
#        np.dstack((label, strength)),
#        window_size=3)
#
#    assert np.allclose(segmentation_slow, segmentation_fast), \
#        "Optimized segmentation is not equivalent to slow version."


shapelist = [(15, 15), (20, 20), (40, 40)]
for shape in shapelist:
    test_growcut(shape)
print("Tests done!")
