import numpy as np
import os
# from skimage.feature import local_binary_pattern

def LBPimage(gray_image):
    h, w = gray_image.shape
    gray_image2 = np.zeros((h+2, w+2))
    gray_image2[1:-1, 1:-1] = gray_image
    imgLBP = np.zeros_like(gray_image)
    neighboor = 3
    for ih in range(h):
        for iw in range(w):
            img = gray_image2[ih:ih + neighboor, iw:iw + neighboor]
            center = img[1, 1]
            img01 = (img >= center) * 1.0
            img01_vector = img01.T.flatten()
            img01_vector = np.delete(img01_vector, 4)
            num = np.sum(img01_vector)
            imgLBP[ih, iw] = num

    return imgLBP


def Lbp_hist(image_gr, cells=2, n_points=1, mode='uniform', n=9):
    rows, cols = image_gr.shape

    cell_x = int(rows / cells)
    cell_y = int(cols / cells)
    lbp_sum = np.array([])
    radius = int(8*n_points)
    for row in range(cells):
        for col in range(cells):
            block = image_gr[(row * cell_x):((row+1) * cell_x), (col * cell_y) : ((col+1) * cell_y)]
            # lbp = local_binary_pattern(block, radius, n_points, mode).reshape(cell_y*cell_x, )
            lbp = LBPimage(block)
            train_hist, _ = np.histogram(lbp, bins=np.arange(n+1),  density=False)
            arr = train_hist / np.sqrt(np.sum(train_hist ** 2) + 1e-6)
            lbp_sum = np.hstack((lbp_sum, arr))

    return lbp_sum

