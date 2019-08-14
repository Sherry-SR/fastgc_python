""" Implementation of the fast grow-cut algorithm with dijkstra """

import numpy as np

from skimage import img_as_float
import nibabel as nib
import matplotlib.pylab as plt
from scipy.ndimage.morphology import binary_dilation, binary_erosion
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

def fastgc(img, seeds, newSeg = True, labPre=None, distPre=None, verbose = True):
    img = img_as_float(img)

    # initialization of current distance and label for growcut
    distCrt = np.zeros(seeds.shape)
    labCrt = np.zeros(seeds.shape)
    if newSeg:
        # for newSeg, use copy seed label as current label
        labCrt = np.copy(seeds)
        # distCrt=0 at seeds, distCrt=np.inf elsewhere
        distCrt[seeds == 0] = np.inf
    else:
        # if not newSeg, only use the seeds with different labels from labPre
        if labPre is None or distPre is None:
            raise Exception("No previous label or distance provided!")
        mask = seeds>0 and seeds != labPre
        # update labCrt with seed label
        labCrt[mask] = seeds[mask]
        distCrt[mask] = 0
        labCrt[~mask] = 0
        distCrt[~mask] = np.inf
        
    # initialzation of fibonacci heap
    fh = FibonacciHeap()
    # choose non-labeled pixels and their neighbors as heap nodes
    mask = labCrt == 0
    mask = binary_dilation(mask)
    iterates = np.array(np.nonzero(mask))
    heapNodes = np.empty(img.shape, dtype = FibonacciHeap.Node)
    # insert to fibonacci heap with key value equal to distCrt(p)
    for i in range(0, iterates.shape[1]):
        ind = tuple(iterates[:,i])
        heapNodes[ind] = fh.insert(distCrt[ind], ind)

    Ntotal = fh.total_nodes
    count = 1
    # segmentation/refinement loop
    while fh.total_nodes > 0:
        pnode = fh.extract_min()
        pind = pnode.value
        # update locally
        if not newSeg:
            if np.isinf(distCrt[pind]):
                break
            # compare current distance with previous distance, use the shortest one
            elif distCrt[pind] > distPre[pind]:
                distCrt[pind] = distPre[pind]
                labCrt[pind] = labPre[pind]
                continue
        # regular dijkstra
        if verbose:
            print("-----------Dijkastra-----------")
            print(str(count)+"/"+str(Ntotal),"Current point:", ind, "Distance from seed:", distCrt[ind], "Seed Label:", labCrt[ind])
            count = count + 1
        neighbours = get_neighbours(np.array(pind), exclude_p=True, shape=img.shape)
        for ind in neighbours:
            ind =tuple(ind)
            dist = distCrt[pind] + np.linalg.norm(img[tuple(ind)] - img[pind])
            if dist < distCrt[ind]:
                distCrt[ind] = dist
                labCrt[ind] = labCrt[pind]
                # update fiponacci heap
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
