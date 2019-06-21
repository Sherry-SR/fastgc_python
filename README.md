# Fast Growcut alogorithm with shortest path 

Implemented fast grow cut algorithm based on "An Effective Interactive Medical Image Segmentation Method Using Fast GrowCut" (see [link](https://nac.spl.harvard.edu/files/nac/files/zhu-miccai2014.pdf))

## Installation
run setup file (currently has some bugs, do not run this step)

```python setup.py build```

```sudo python setup.py install```

## Usage
run fastgc function with image data **img**, initial seed labels **seeds**

```fastgc(img, seeds, newSeg = True, labCrt=None, distCrt=None, labPre=None, distPre=None, verbose = True)```

## Test example
run ```pytest``` directly

### Test list

Test 1: **test_growcut.py**

