import numpy as np

from scipy import fftpack
from numpy import r_

def dct2(a):
    return fftpack.dct( fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return fftpack.idct( fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def transform(X, a, b, r):
    if(r >= a*b): # 24 >= 4*6
        x_dcc = np.zeros(X.shape)
        h = X.shape[0]
        w = X.shape[1]
        thresh_flex = (r-a*b)/h/w
        for d in range(X.shape[2]):
            im = X[:,:,d]
            imsize = im.shape
            dct = np.zeros(imsize)
            dcc = np.zeros(imsize)
            for i in range(a):
                for j in range(b):
                    start_h, end_h = int(h/a*i), int(h/a*(i+1))
                    start_w, end_w = int(w/b*j), int(w/b*(j+1))
                    dct[start_h:end_h, start_w:end_w] = dct2(im[start_h:end_h, start_w:end_w])
                    max_idx = np.argmax(dct[start_h:end_h, start_w:end_w])
                    max_point = divmod(max_idx, end_w-start_w)
                    dcc[start_h+max_point[0], start_w+max_point[1]] = dct[start_h+max_point[0], start_w+max_point[1]]
                    dct[start_h+max_point[0], start_w+max_point[1]] = 0
            over_threshold = abs(dct) > (np.sort(abs(dct.flatten()))[::-1][int(thresh_flex * len(dct.flatten()))])
            dcc = dcc + dct * over_threshold
            im_dct = np.zeros(imsize)
            for i in range(a):
                for j in range(b):
                    start_h, end_h = int(h/a*i), int(h/a*(i+1))
                    start_w, end_w = int(w/b*j), int(w/b*(j+1))
                    im_dct[start_h:end_h, start_w:end_w] = idct2(dcc[start_h:end_h, start_w:end_w])
            x_dcc[:,:,d] = np.clip(im_dct, 0, 1)
        return x_dcc
    else:
        raise Exception("r must be equal or greater than a*b")