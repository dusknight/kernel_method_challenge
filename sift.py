# Adapted from OpenCV, original comments are conserved
# see `opencv/modules/features2d/src/sift.simd.hpp`
# and `opencv/modules/features2d/src/sift.dispatch.cpp`
import numpy as np
from copy import deepcopy

EPS = 1e-7
SIFT_INIT_SIGMA = 0.5
SIFT_ORI_HIST_BINS = 8
SIFT_IMG_BORDER = 4
SIFT_MAX_INTERP_STEPS = 5
SIFT_ORI_SIG_FCTR = 1.5
SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR
SIFT_ORI_PEAK_RATIO = 0.8
SIFT_DESCR_WIDTH = 2
SIFT_DESCR_HIST_BINS = 8
SIFT_DESCR_SCL_FCTR = 3.
SIFT_DESCR_MAG_THR = 0.2


def _create_gaussian_kernel(sigma, kernel_size):
    # see `opencv2/imgproc.hpp`: getGaussianKernel()
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i < (kernel_size + 1) / 2 and j < (kernel_size + 1) / 2:
                kernel[i, j] = np.exp(
                    -((i + 0.5 - 0.5 * kernel_size) ** 2 + (j + 0.5 - 0.5 * kernel_size) ** 2) / (2 * sigma ** 2))
            elif j >= (kernel_size + 1) / 2:
                kernel[i, j] = kernel[i, kernel_size - 1 - j]
            else:
                kernel[i, j] = kernel[kernel_size - 1 - i, kernel_size - 1 - j]
    kernel /= np.sum(kernel)
    return kernel


def gaussian_blur(I, sigma, kernel_size=None):
    # see `opencv2/imgproc.hpp`: GaussianBlur()
    if kernel_size is None:
        kernel_size = int(np.floor(sigma * 3)) * 2 + 1

    nx = I.shape[0]
    ny = I.shape[1]
    kernel = _create_gaussian_kernel(sigma, kernel_size)
    kernel_center = (kernel_size - 1) // 2
    new_I = np.zeros((nx, ny))
    for x in range(nx):
        for y in range(ny):
            if (x - kernel_center < 0) or (x + kernel_center >= nx) or (y - kernel_center > 0) or (
                    y + kernel_center >= ny):
                kernel_sum = 0
                for dx in range(-kernel_center, kernel_center + 1):
                    for dy in range(-kernel_center, kernel_center + 1):
                        if x + dx >= 0 and x + dx < nx and y + dy >= 0 and y + dy < ny:
                            new_I[x, y] += kernel[kernel_center + dx, kernel_center + dy] * I[x + dx, y + dy]
                            kernel_sum += kernel[kernel_center + dx, kernel_center + dy]
                new_I[x, y] /= kernel_sum
            else:
                for dx in range(-kernel_center, kernel_center + 1):
                    for dy in range(-kernel_center, kernel_center + 1):
                        new_I[x, y] += kernel[kernel_center + dx, kernel_center + dy] * I[x + dx, y + dy]
    return new_I


def linear_sample_image(I, x, y):
    # see `cv2.resize()`
    left = min(max(int(np.floor(x)), 0), I.shape[0] - 2)
    top = min(max(int(np.floor(y)), 0), I.shape[1] - 2)
    wx = x - left
    wy = y - top
    return (I[left, top] * (1 - wx) + I[left + 1, top] * wx) * (1 - wy) + (
            I[left, top + 1] * (1 - wx) + I[left + 1, top + 1] * wx) * wy


def inverse_transform_linear(I, sizeX, sizeY, scale, rotation, translateX, translateY):
    # see `cv2.resize()`
    result = np.empty((sizeX, sizeY))
    for x in range(sizeX):
        for y in range(sizeY):
            cos = np.cos(rotation)
            sin = np.sin(rotation)
            x2 = x * scale
            y2 = y * scale
            x3 = x2 * cos - y2 * sin + translateX
            y3 = y2 * cos + x2 * sin + translateY
            result[x, y] = linear_sample_image(I, x3, y3)
    return result


class Keypoint:  # data class for keypoints detected in SIFT
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.octave = 0
        self.layer = 0.
        self.sigma = 0.
        self.angle = 0.
        self.importance = 0


class SIFT:
    """
    for detailed reference, see https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html
    """
    def __init__(self, nfeatures=10, nOctaveLayers=3, contrastThreshold=0.001, edgeThreshold=10, sigma=1.):
        """
        create SIFT object
        :param nfeatures: number of best features to retain.
        :param nOctaveLayers: number of layers in each octave
        :param contrastThreshold: used to filter out weak features in semi-uniform (low-contrast) regions
        :param edgeThreshold: used to filter out edge-like features
        :param sigma: applied to the input image at the octave.
            If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        """
        self.nfeatures = nfeatures
        self.noctaves = -1
        self.noctave_layers = nOctaveLayers
        self.contrast_threshold = contrastThreshold
        self.edge_threshold = edgeThreshold
        self.sigma = sigma
        self.base_image = None
        self.gaussian_pyramid = None
        self.dog_pyramid = None
        self.keypoints = None
        self.descriptors = None

    def _create_initial_image(self, grayI):
        """
        We only accept gray scale image, should be (HxW)
        :param grayI: image 2D array
        :return: None
        """
        # Double the size
        result = inverse_transform_linear(grayI, grayI.shape[0] * 2, grayI.shape[1] * 2, 0.5, 0, 0, 0)

        sig_diff = np.sqrt(max(self.sigma * self.sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01))
        self.base_image = gaussian_blur(result, sig_diff)

    def _build_gaussian_pyramid(self):
        sig = np.empty(self.noctave_layers + 3)
        # self.gaussian_pyramid = np.empty(self.noctaves * (self.noctave_layers + 3))
        self.gaussian_pyramid = []

        sig[0] = self.sigma
        k = np.power(2., 1. / self.noctave_layers)
        for i in range(1, self.noctave_layers + 3):
            sig_prev = np.power(k, i - 1) * self.sigma
            sig_total = sig_prev * k
            sig[i] = np.sqrt(sig_total ** 2 - sig_prev ** 2)

        for o in range(self.noctaves):
            for i in range(self.noctave_layers + 3):
                # dst = o * (self.noctave_layers + 3) + i
                if o == 0 and i == 0:
                    self.gaussian_pyramid.append(self.base_image)
                # base of new octave is halved image from end of previous octave
                elif i == 0:
                    src = (o - 1) * (self.noctave_layers + 3) + self.noctave_layers
                    self.gaussian_pyramid.append(inverse_transform_linear(
                        self.gaussian_pyramid[src],
                        self.gaussian_pyramid[src].shape[0] // 2,
                        self.gaussian_pyramid[src].shape[1] // 2,
                        2, 0, 0, 0))
                else:
                    src = o * (self.noctave_layers + 3) + i - 1
                    self.gaussian_pyramid.append(gaussian_blur(self.gaussian_pyramid[src], sig[i]))

    def _build_DoG_pyramid(self):
        self.dog_pyramid = []

        for o in range(self.noctaves):
            for i in range(self.noctave_layers + 2):
                src1 = o * (self.noctave_layers + 3) + i
                src2 = src1 + 1
                # dst = o * (self.noctave_layers + 2) + i
                self.dog_pyramid.append(self.gaussian_pyramid[src2] - self.gaussian_pyramid[src1])

    # Computes a gradient orientation histogram at a specified pixel
    def _calc_orientation_hist(self, I, px, py, radius, weight_sigma, nbins):
        expf_scale = -1. / (2. * weight_sigma * weight_sigma)
        width = I.shape[0]
        height = I.shape[1]
        temphist = np.zeros(nbins)

        for i in range(-radius, radius + 1):
            x = px + i
            if x <= 0 or x >= width - 1:
                continue
            for j in range(-radius, radius + 1):
                y = py + j
                if y <= 0 or y >= height - 1:
                    continue

                dx = I[x + 1, y] - I[x - 1, y]
                dy = I[x, y + 1] - I[x, y - 1]

                # compute gradient values, orientations and the weights over the pixel neighborhood
                weight = np.exp((i * i + j * j) * expf_scale)
                angle = np.arctan2(dy, dx)
                mag = np.sqrt(dx * dx + dy * dy)

                binnum = int(np.round((nbins / (2 * np.pi)) * angle))
                if binnum >= nbins:
                    binnum -= nbins
                if binnum < 0:
                    binnum += nbins
                temphist[binnum] += weight * mag

        # smooth the histogram
        hist = np.zeros(nbins)
        for i in range(nbins):
            hist[i] = (temphist[(i - 2 + nbins) % nbins] + temphist[(i + 2) % nbins]) * (1. / 16.) + \
                      (temphist[(i - 1 + nbins) % nbins] + temphist[(i + 1) % nbins]) * (1. / 4.) + \
                      temphist[i] * (6. / 16.)

        return hist

    # Interpolate location of extrama, scale to subpixel accuracy, rejects features with low contrast.
    def _adjust_local_extrema(self, octv, layer, x, y):
        di, dx, dy = 0, 0, 0
        finished = False

        for _ in range(SIFT_MAX_INTERP_STEPS):
            idx = octv * (self.noctave_layers + 2) + layer
            img = self.dog_pyramid[idx]
            prv = self.dog_pyramid[idx - 1]
            nxt = self.dog_pyramid[idx + 1]

            dD = np.array([(img[x + 1, y] - img[x - 1, y]) * 0.5, (img[x, y + 1] - img[x, y - 1]) * 0.5,
                              (nxt[x, y] - prv[x, y]) * 0.5])

            v2 = img[x, y] * 2
            dxx = img[x + 1, y] + img[x - 1, y] - v2
            dyy = img[x, y + 1] + img[x, y - 1] - v2
            dss = nxt[x, y] + prv[x, y] - v2
            dxy = (img[x + 1, y + 1] - img[x + 1, y - 1] - img[x - 1, y + 1] + img[x - 1, y - 1]) * 0.25
            dxs = (nxt[x + 1, y] - nxt[x - 1, y] - prv[x + 1, y] + prv[x - 1, y]) * 0.25
            dys = (nxt[x, y + 1] - nxt[x, y - 1] - prv[x, y + 1] + prv[x, y - 1]) * 0.25

            H = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
            try:
                X = np.linalg.solve(H, dD)
            except np.linalg.LinAlgError as e:
                finished = True
                break
            dx = -X[0]
            dy = -X[1]
            di = -X[2]

            if abs(dx) < 0.5 and abs(dy) < 0.5 and abs(di) < 0.5:
                finished = True
                break

            x += int(np.round(dx))
            y += int(np.round(dy))
            layer += int(np.round(di))

            if (layer < 1 or layer > self.noctave_layers or
                    x < SIFT_IMG_BORDER or x >= img.shape[0] - SIFT_IMG_BORDER or
                    y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER):
                return None, 0, 0, 0

        # ensure convergence of interpolation
        if not finished:
            return None, 0, 0, 0

        idx = octv * (self.noctave_layers + 2) + layer
        img = self.dog_pyramid[idx]
        prv = self.dog_pyramid[idx - 1]
        nxt = self.dog_pyramid[idx + 1]
        dD = np.array([(img[x + 1, y] - img[x - 1, y]) * 0.5, (img[x, y + 1] - img[x, y - 1]) * 0.5,
                          (nxt[x, y] - prv[x, y]) * 0.5])
        t = np.dot(dD, np.array([dx, dy, di]))

        contr = img[x, y] + t * 0.5
        if abs(contr) * self.noctave_layers < self.contrast_threshold:
            return None, 0, 0, 0

        # principal curvatures are computed using the trace and det of Hessian
        v2 = img[x, y] * 2
        dxx = img[x + 1, y] + img[x - 1, y] - v2
        dyy = img[x, y + 1] + img[x, y - 1] - v2
        dxy = (img[x + 1, y + 1] - img[x + 1, y - 1] - img[x - 1, y + 1] + img[x - 1, y - 1]) * 0.25
        tr = dxx + dyy
        det = dxx * dyy - dxy * dxy

        if det <= 0 or tr * tr * self.edge_threshold >= ((self.edge_threshold + 1) ** 2) * det:
            return None, 0, 0, 0

        kpt = Keypoint()
        kpt.x = (x + dx) * (1 << octv)
        kpt.y = (y + dy) * (1 << octv)
        kpt.octave = octv
        kpt.layer = layer + di
        kpt.sigma = self.sigma * np.power(2.0, (layer + di) / self.noctave_layers) * (1 << octv)
        kpt.importance = abs(contr)
        return kpt, layer, x, y

    def _find_scale_space_extrema(self):
        threshold = 0.5 * self.contrast_threshold / self.noctave_layers

        self.keypoints = []

        ncandidate = 0

        for o in range(self.noctaves):
            for i in range(1, self.noctave_layers + 1):
                idx = o * (self.noctave_layers + 2) + i
                img = self.dog_pyramid[idx]
                prv = self.dog_pyramid[idx - 1]
                nxt = self.dog_pyramid[idx + 1]
                width = img.shape[0]
                height = img.shape[1]

                for x in range(SIFT_IMG_BORDER, width - SIFT_IMG_BORDER):
                    for y in range(SIFT_IMG_BORDER, height - SIFT_IMG_BORDER):
                        val = img[x, y]
                        # find local extrema with pixel accuracy
                        if abs(val) > threshold and ((val > 0 and val >= img[x, y - 1] and val >= img[
                            x, y + 1] and
                                                      val >= img[x + 1, y] and val >= img[x - 1, y] and
                                                      val >= nxt[x, y] and val >= prv[x, y]) or (
                                                             val < 0 and val <= img[x, y - 1] and val <= img[
                                                         x, y + 1] and
                                                             val <= img[x + 1, y] and val <= img[x - 1, y] and
                                                             val <= nxt[x, y] and val <= prv[x, y])):

                            ncandidate += 1
                            kpt, i2, x2, y2 = self._adjust_local_extrema(o, i, x, y)
                            if kpt is None:
                                continue
                            scl_octv = kpt.sigma / (1 << o)
                            n = SIFT_ORI_HIST_BINS
                            hist = self._calc_orientation_hist(
                                self.gaussian_pyramid[o * (self.noctave_layers + 3) + i2],
                                x2,
                                y2,
                                int(np.round(SIFT_ORI_RADIUS * scl_octv)),
                                SIFT_ORI_SIG_FCTR * scl_octv, n)

                            mag_threshold = np.max(hist) * SIFT_ORI_PEAK_RATIO
                            for j in range(n):
                                left = j - 1 if j > 0 else n - 1
                                right = j + 1 if j < n - 1 else 0

                                if hist[j] > hist[left] and hist[j] > hist[right] and hist[j] >= mag_threshold:
                                    binnum = j + 0.5 * (hist[left] - hist[right]) / (
                                            hist[left] - 2 * hist[j] + hist[right])
                                    binnum = binnum + n if binnum < 0 else binnum
                                    binnum = binnum - n if binnum >= n else binnum
                                    kpt.angle = (2 * np.pi / n) * binnum
                                    self.keypoints.append(deepcopy(kpt))

    def _calc_SIFT_descriptor(self, I, xf, yf, angle, sigma):
        d = SIFT_DESCR_WIDTH
        n = SIFT_DESCR_HIST_BINS
        x = int(np.round(xf))
        y = int(np.round(yf))
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        bins_per_rad = n / (2 * np.pi)
        exp_scale = -1. / (d * d * 0.5)
        hist_width = SIFT_DESCR_SCL_FCTR * sigma
        radius = int(np.round(hist_width * 1.4142135623730951 * (d + 1) * 0.5))
        cos_t /= hist_width
        sin_t /= hist_width

        width = I.shape[0]
        height = I.shape[1]

        hist = np.zeros((d, d, n))

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                # Calculate sample's histogram array coords rotated relative to ori.
                # Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
                # r_rot = 1.5) have full weight placed in row 1 after interpolation.
                x_rot = i * cos_t + j * sin_t
                y_rot = j * cos_t - i * sin_t
                xbin = x_rot + d / 2 - 0.5
                ybin = y_rot + d / 2 - 0.5
                xt = x + i
                yt = y + j

                if 0 <= xbin < d - 1 and ybin >= 0 and ybin < d - 1 and xt > 0 and xt < width - 1 and yt > 0 and yt < height - 1:
                    dx = I[x + 1, y] - I[x - 1, y]
                    dy = I[x, y + 1] - I[x, y - 1]
                    grad_angle = np.arctan2(dy, dx)
                    grad_mag = np.sqrt(dx * dx + dy * dy) * np.exp((x_rot * x_rot + y_rot * y_rot) * exp_scale)
                    obin = (grad_angle - angle) * bins_per_rad
                    x0 = int(np.floor(xbin))
                    y0 = int(np.floor(ybin))
                    o0 = int(np.floor(obin))
                    xbin -= x0
                    ybin -= y0
                    obin -= o0
                    if o0 < 0:
                        o0 += n
                    if o0 >= n:
                        o0 -= n

                    # histogram update using tri-linear interpolation
                    v_x1 = grad_mag * xbin
                    v_x0 = grad_mag - v_x1
                    v_xy11 = v_x1 * ybin
                    v_xy10 = v_x1 - v_xy11
                    v_xy01 = v_x0 * ybin
                    v_xy00 = v_x0 - v_xy01
                    v_xyo111 = v_xy11 * obin
                    v_xyo110 = v_xy11 - v_xyo111
                    v_xyo101 = v_xy10 * obin
                    v_xyo100 = v_xy10 - v_xyo101
                    v_xyo011 = v_xy01 * obin
                    v_xyo010 = v_xy01 - v_xyo011
                    v_xyo001 = v_xy00 * obin
                    v_xyo000 = v_xy00 - v_xyo001

                    hist[x0, y0, o0] += v_xyo000
                    hist[x0, y0, (o0 + 1) % n] += v_xyo001
                    hist[x0, y0 + 1, o0] += v_xyo010
                    hist[x0, y0 + 1, (o0 + 1) % n] += v_xyo011
                    hist[x0 + 1, y0, o0] += v_xyo100
                    hist[x0 + 1, y0, (o0 + 1) % n] += v_xyo101
                    hist[x0 + 1, y0 + 1, o0] += v_xyo110
                    hist[x0 + 1, y0 + 1, (o0 + 1) % n] += v_xyo111

        # copy histogram to the descriptor,
        # apply hysteresis thresholding
        # and scale the result, so that it can be easily converted
        # to byte array
        hist = hist.flatten()
        nrm2 = np.linalg.norm(hist) **2
        threshold = np.sqrt(nrm2) * SIFT_DESCR_MAG_THR
        hist = np.where(hist < threshold, hist, threshold)
        # for i in range(len(hist)):
        #     hist[i] = np.min(hist[i], threshold)
        nrm2_sqrt = np.linalg.norm(hist)
        # print(nrm2_sqrt, max(nrm2_sqrt, EPS))
        hist /= max(nrm2_sqrt, EPS)
        return hist

    def _calc_descriptors(self, unflatten):
        ret = []
        for i in range(len(self.keypoints)):
            kpt = self.keypoints[i]
            assert kpt.octave >= -1 and kpt.layer <= self.noctave_layers + 2
            scale = 1 / np.exp2(kpt.octave)
            size = kpt.sigma * scale
            img = self.gaussian_pyramid[(kpt.octave + 1) * (self.noctave_layers + 3) + int(np.round(kpt.layer))]
            ret.append(self._calc_SIFT_descriptor(img, kpt.x * scale, kpt.y * scale, kpt.angle, size))
        if unflatten:
            return np.array(ret)
        return np.array(ret).flatten()

    def calc_features_for_image(self, I, unflatten):
        self.noctaves = int(np.round(np.log2(min(I.shape[0], I.shape[1])))) - 1
        self._create_initial_image(I)
        self._build_gaussian_pyramid()
        self._build_DoG_pyramid()
        self._find_scale_space_extrema()

        kpt = Keypoint()
        kpt.x, kpt.y = 16, 16
        kpt.octave = 1
        kpt.sigma = self.sigma * np.power(2.0, kpt.layer / self.noctave_layers) * (1 << kpt.octave)
        self.keypoints.append(kpt)

        self.keypoints.sort(key=lambda kpt: kpt.importance, reverse=True)
        # remove duplicate
        filtered_keypoints = [self.keypoints[0]]
        for i in range(1, len(self.keypoints)):
            if self.keypoints[i].x != self.keypoints[i - 1].x or \
                    self.keypoints[i].y != self.keypoints[i - 1].y or \
                    self.keypoints[i].sigma != self.keypoints[i - 1].sigma or \
                    self.keypoints[i].angle != self.keypoints[i - 1].angle:
                filtered_keypoints.append(self.keypoints[i])
        # retain best
        if len(self.keypoints) > self.nfeatures:
            self.keypoints = filtered_keypoints[:self.nfeatures]
        elif not unflatten:
            for i in range(len(self.keypoints), self.nfeatures):
                self.keypoints.append(deepcopy(self.keypoints[0]))

        for kpt in self.keypoints:
            kpt.octave -= 1
            kpt.x /= 2
            kpt.y /= 2
            kpt.sigma /= 2

        return self._calc_descriptors(unflatten)
