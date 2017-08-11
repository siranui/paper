from skimage import io
import numpy as np
import sys

args = sys.argv
fn = sys.argv[1]
gn = fn.split('/')[-1] + '.png'

rows = 0
cols = 0
if len(args) == 4:
    rows = int(args[2])
    cols = int(args[3])
elif len(args) == 3:
    rows = cols = int(args[2])
else:
    rows = cols = 1

w = 32
h = 32

ims = np.zeros([rows*h,cols*w],dtype=np.uint8)
lines = open(fn).readlines()
for r in range(rows):
  for c in range(cols):
    line = lines[r * cols + c]
    im = np.array([float(i) for i in line.strip().split(',')]).reshape(h,w)
    imax = max(im.reshape(w*h))
    imin = min(im.reshape(w*h))
    iran = imax - imin
    def cc(x):
      a = int(((x - imin) / iran) * 256)
      return 0 if a < 0 else 255 if a > 255 else a

    for i in range(h):
      for j in range(w):
        ims[r * h + i, c * w + j] = cc(im[i,j])

io.imsave(gn, ims)
