""" Implementation of the fast grow-cut algorithm with dijkstra """

import numpy as np

from skimage import img_as_float
import nibabel as nib
import matplotlib.pylab as plt
from skimage.morphology import dilation, erosion, square
from math import sqrt
from myutils.fibheap import FibonacciHeap

def get_neighbours(p, exclude_p=True, shape=None):
    ndim = len(p)
    # generate an (m, ndims) array containing all strings over the alphabet {0, 1, 2}:
    offset_idx = np.indices((3,) * ndim).reshape(ndim, -1).T
    # use these to index into np.array([-1, 0, 1]) to get offsets
    offsets = np.r_[-1, 0, 1].take(offset_idx)
    # optional: exclude offsets of 0, 0, ..., 0 (i.e. p itself)
    if exclude_p:
        offsets = offsets[np.any(offsets, 1)]

    neighbours = p + offsets    # apply offsets to p
    # optional: exclude out-of-bounds indices
    if shape is not None:
        valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
        neighbours = neighbours[valid]
    return neighbours

def fastgc(img, seeds, newSeg, labCrt=None, distCrt=None, labPre=None, distPre=None):
    img = img_as_float(img)

    # initialization of current distance and label for growcut
    distCrt = np.zeros(img.shape)
    if newSeg:
        labCrt = np.copy(seeds)
        distCrt[seeds == 0] = np.inf
    else:
        mask = seeds>0 and seeds != labPre
        labCrt[mask] = seeds[mask]
        distCrt[mask] = 0
        labCrt[~mask] = 0
        distCrt[~mask] = np.inf
        
    # initialzation of fibonacci heap
    fh = FibonacciHeap()
    iterates = np.indices(img.shape)
    iterates = np.reshape(iterates, (iterates.shape[0],-1))
    heapNodes = np.empty(img.shape, dtype = FibonacciHeap.Node)
    for i in range(0, img.size):
        ind = tuple(iterates[:,i])
        heapNodes[ind] = fh.insert(distCrt[ind], ind)

    # segmentation/refinement loop
    while fh.total_nodes > 0:
        pnode = fh.extract_min()
        pind = pnode.value
        # update locally
        if not newSeg:
            if np.isinf(distCrt[pind]):
                break
            elif distCrt[pind] > distPre[pind]:
                distCrt[pind] = distPre[pind]
                labCrt[pind] = labPre[pind]
                continue
        # regular dijkstra
        print("-----------Dijkastra-----------")
        print("Current point:", ind, "Distance from seed:", distCrt[ind], "Seed Label:", labCrt[ind])
        neighbours = get_neighbours(np.array(pind), exclude_p=True, shape=img.shape)
        for ind in neighbours:
            ind =tuple(ind)
            dist = distCrt[pind] + np.linalg.norm(img[tuple(ind)] - img[pind])
            if dist < distCrt[ind]:
                distCrt[ind] = dist
                labCrt[ind] = labCrt[pind]
                fh.decrease_key(heapNodes[ind], distCrt[ind])
    if not newSeg:
        # get updated points
        mask = ~np.isinf(distCrt)
        # update local states        
        labPre[mask] = labCrt[mask]
        distPre[mask] = distCrt[mask]
        labCrt = np.copy(labPre)
        distCrt = np.copy(distPre)
    # save current results
    distPre = np.copy(distCrt)
    labPre = np.copy(labCrt)
    return distPre, labPre
        
img = nib.load("/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001.nii.gz")
imgdata = img.get_fdata()[420:500, 130:220, 226]
label = nib.load("/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001_label_tri.nii.gz")
labeldata = label.get_fdata()[420:500, 130:220, 226]
seedsdata = np.zeros(labeldata.shape)

nlabels = np.unique(labeldata)
for i in nlabels:
    mask = labeldata == i
    mask = erosion(mask, square(3))
    mask = erosion(mask, square(3))
    seedsdata = seedsdata + mask * (i+1)

distPre, labPre = fastgc(imgdata, seedsdata, True)
plt.figure(1)
plt.imshow(imgdata, cmap="gray")
plt.title("original img")
plt.figure(2)
plt.imshow(labeldata)
plt.title("original label")
plt.figure(3)
plt.imshow(seedsdata)
plt.title("original seeds")
plt.figure(4)
plt.imshow(labPre)
plt.title("growcut results")

plt.show()

