#  Fast Growcut alogorithm with shortest path 

Implemented fast grow cut algorithm based on "An Effective Interactive Medical Image Segmentation Method Using Fast GrowCut" (see [link](https://nac.spl.harvard.edu/files/nac/files/zhu-miccai2014.pdf)), generalized for multi-class n-dimensional data.

## Installation
Run setup file (ONLY needed for Cython version, NOT required for the other Python implementation)

```python setup.py build```

```sudo python setup.py install```

## Usage
For basic GrowCut package usase, see ([link](https://github.com/nfaggian/growcut)).

**Fast Growcut using shortest path**

Implemented in [growcut_cpu](./modules/growcut_cpu.py)

Run fastgc function with image data **img**, initial seed labels **seeds**, both img and seeds  can be n-dimensional data.

```fastgc(img, seeds, newSeg = True, labCrt=None, distCrt=None, labPre=None, distPre=None, verbose = True)```

## Test example
Run ```pytest``` directly

### Test list

Test 1: **test_growcut.py**

### Github reference

1. Basic GrowCut package ([link](https://github.com/nfaggian/growcut))
2. Fibonacci heaps package ([link](https://github.com/danielborowski/fibonacci-heap-python))