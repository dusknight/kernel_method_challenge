import numpy as np

# compute gradient of an image
def image_grad(image_array, max_h, max_w):
    grad = np.zeros([max_h, max_w])
    mag = np.zeros([max_h, max_w])
    for h, row in enumerate(image_array):
        for w, val in enumerate(row):
            if h - 1 >= 0 and w - 1 >= 0 and h + 1 < max_h and w + 1 < max_w:
                dy = image_array[h + 1][w] - image_array[h - 1][w]
                dx = row[w + 1] - row[w - 1] + 1e-5
                grad[h][w] = np.arctan(dy / dx) * (180 / np.pi)
                if grad[h][w] < 0:
                    grad[h][w] += 180
                mag[h][w] = np.sqrt(dy * dy + dx * dx)

    return grad, mag


def div_cells(mag, cell_x, cell_y, cell_w):
    mag_cells = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    mag_x = np.split(mag, cell_x, axis=0)
    for i, l in enumerate(mag_x):
        mag_x[i] = np.array(l)
        mag_xy = np.split(mag_x[i], cell_y, axis=1)
        for j, l1 in enumerate(mag_xy):
            mag_xy[j] = np.array(l1)
            mag_cells[i][j] = mag_xy[j]

    return mag_cells


def get_bins(mag_cell, ang_cell):
    bin_num = 9
    bins = [0.0] * bin_num
    offset = 20

    mag_list = mag_cell.flatten()
    ang_list = ang_cell.flatten()

    for i, ang in enumerate(ang_list):
        if ang >= 180:
            ang -= 180

        left_bin = int(ang / offset)

        right_bin = left_bin + 1 if left_bin != bin_num - 1 else 0

        right_ratio = ang / offset - left_bin
        left_ration = 1 - right_ratio

        bins[left_bin] += mag_list[i] * left_ration
        bins[right_bin] += mag_list[i] * right_ratio

    return bins


def hog(img, im_size=(32,32), cell_size=4, overlap=True, clip=None):
    x_pixel = im_size[0]
    y_pixel = im_size[1]

    cell_x = int(x_pixel / cell_size)
    cell_y = int(y_pixel / cell_size)

    ang, mag = image_grad(img, x_pixel, y_pixel)
    mag_cells = div_cells(mag, cell_x, cell_y, cell_size)
    ang_cells = div_cells(ang, cell_x, cell_y, cell_size)
    hog_descriptor = np.array([])

    for x in range(cell_x - 1):
        for y in range(cell_y - 1):
            hist = []
            hist.extend(get_bins(mag_cells[x][y], ang_cells[x][y]))

            if overlap:
                hist.extend(get_bins(mag_cells[x][y + 1], ang_cells[x][y + 1]))
                hist.extend(get_bins(mag_cells[x + 1][y], ang_cells[x + 1][y]))
                hist.extend(get_bins(mag_cells[x + 1][y + 1], ang_cells[x + 1][y + 1]))

            arr = np.array(hist)
            if clip is not None:
                arr = np.clip(arr, 0, clip)
            arr = arr / np.sqrt(np.sum(arr ** 2) + 1e-5 ** 2)
            hog_descriptor = np.hstack((hog_descriptor, arr))
    return hog_descriptor

# Todo: change descriptor length? add padding?
