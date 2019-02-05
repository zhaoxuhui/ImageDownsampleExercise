# coding=utf-8
from numba import cuda
import numpy as np
import cv2
import time
import math


@cuda.jit
def downSampleGPU(inImg, downSize, outImg):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    x = bx * bw + tx
    y = by * bh + ty

    if x < inImg.shape[0] and y < inImg.shape[1]:
        if x / downSize < outImg.shape[0] and y / downSize < outImg.shape[1]:
            outImg[x / downSize, y / downSize] = inImg[x, y]


if __name__ == '__main__':

    img = cv2.imread("test.tif", cv2.IMREAD_GRAYSCALE)

    down_sample = 2
    height = img.shape[0]
    width = img.shape[1]

    new_height = height / down_sample
    new_width = width / down_sample

    img_new = np.zeros([new_height, new_width], img.dtype)

    THREAD_SIZE = 32

    threadsperblock = (THREAD_SIZE, THREAD_SIZE)
    blockspergrid = (int(math.ceil(img.shape[0] * 1.0 / THREAD_SIZE)),
                     int(math.ceil(img.shape[1] * 1.0 / THREAD_SIZE)))

    print 'matrix size:', img.shape
    print 'thread per block:', threadsperblock
    print 'block per grid:', blockspergrid

    for i in range(100):
        t1 = time.time()
        downSampleGPU[blockspergrid, threadsperblock](img, down_sample, img_new)
        t2 = time.time()
        print t2 - t1
    # cv2.imwrite("res.jpg", img_new)
