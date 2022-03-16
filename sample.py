import numpy as np
import sklearn
import cv2

from dcc import transform

HIGHT = 218
WIDTH = 178
a, b, r = 4, 4, 64

# Read Image
sample = cv2.imread('./sample.jpg')

# Image to Numpy 
sample = np.array(sample)

# Normalize Sample Image
sample = cv2.normalize(sample/1.0, sample, 0, 1, cv2.NORM_MINMAX)

# Resize Image
sample = cv2.resize(sample, dsize=(WIDTH,HIGHT))

# Transform Image by DCC
sample_dcc = transform(sample, a, b, r)

# Save Result of DCC
cv2.imwrite('result.jpg', sample_dcc*255)
