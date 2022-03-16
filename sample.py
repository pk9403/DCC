import numpy as np
import sklearn
import cv2
import argparse

from dcc import transform

HIGHT = 218
WIDTH = 178

parser = argparse.ArgumentParser()
parser.add_argument("--a", type=int, default=4, help='number of row blocks')
parser.add_argument("--b", type=int, default=4, help='number of column blocks')
parser.add_argument("--r", type=int, default=64, help='number of remaining coefficients')
args = parser.parse_args()

# Read Image
sample = cv2.imread('./sample.jpg')

# Image to Numpy 
sample = np.array(sample)

# Normalize Sample Image
sample = cv2.normalize(sample/1.0, sample, 0, 1, cv2.NORM_MINMAX)

# Resize Image
sample = cv2.resize(sample, dsize=(WIDTH,HIGHT))

# Transform Image by DCC
sample_dcc = transform(sample, args.a, args.b, args.r)

# Save Result of DCC
cv2.imwrite('result.jpg', sample_dcc*255)
