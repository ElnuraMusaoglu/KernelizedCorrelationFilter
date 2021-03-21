'''

Elnura Musaoglu
2021

'''

import numpy as np
import cv2
from numpy.fft import fftn, ifftn, fft2, ifft2, fftshift
from numpy import conj, real
from utils import gaussian2d_rolled_labels, cos_window
from hog_cpp.fhog.get_hog import get_hog

vgg_path = 'model/imagenet-vgg-verydeep-19.mat'

def create_model():
    from scipy import io
    from keras.applications.vgg19 import VGG19
    from keras.models import Model

    mat = io.loadmat(vgg_path)
    model = VGG19(mat)
    ixs = [2, 5, 10, 15, 20]
    outputs = [model.layers[i].output for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)
    # model.summary()
    return model

vgg_model = create_model()

class KernelizedCorrelationFilter:
    def __init__(self, correlation_type='gaussian', feature='hog'):
        self.padding = 1.5  # extra area surrounding the target #padding = 2   #extra area surrounding the target
        self.lambda_ = 1e-4  # regularization
        self.output_sigma_factor = 0.1  # spatial bandwidth (proportional to target)
        self.correlation_type = correlation_type
        self.feature = feature
        self.resize = False
        # GRAY
        if feature == 'gray':
            self.interp_factor = 0.075  # linear interpolation factor for adaptation
            self.sigma = 0.2  # gaussian kernel bandwidth
            self.poly_a = 1  # polynomial kernel additive term
            self.poly_b = 7  # polynomial kernel exponent
            self.gray = True
            self.cell_size = 1
        # HOG
        elif feature == 'hog':
            self.interp_factor = 0.02  # linear interpolation factor for adaptation
            self.sigma = 0.5  # gaussian kernel bandwidth
            self.poly_a = 1  # polynomial kernel additive term
            self.poly_b = 9  # polynomial kernel exponent
            self.hog = True
            self.hog_orientations = 9
            self.cell_size = 4
        # DEEP
        elif feature == 'deep':
            self.interp_factor = 0.02  # linear interpolation factor for adaptation
            self.sigma = 0.5  # gaussian kernel bandwidth
            self.poly_a = 1  # polynomial kernel additive term
            self.poly_b = 9  # polynomial kernel exponent
            self.deep = True
            self.cell_size = 4  # 8

    def start(self, init_gt, show, frame_list):
        poses = []
        poses.append(init_gt)
        init_frame = cv2.imread(frame_list[0])
        x1, y1, w, h = init_gt
        init_gt = tuple(init_gt)
        self.init(init_frame, init_gt)

        for idx in range(len(frame_list)):
            if idx != 0:
                current_frame = cv2.imread(frame_list[idx])
                bbox = self.update(current_frame)
                if bbox is not None:
                    x1, y1, w, h = bbox
                    if show is True:
                        if len(current_frame.shape) == 2:
                            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
                        show_frame = cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                                   (255, 0, 0), 1)
                        cv2.imshow('demo', show_frame)
                        cv2.waitKey(1)
                else:
                    print('bbox is None')
                poses.append(np.array([int(x1), int(y1), int(w), int(h)]))

        return np.array(poses)

    def init(self, image, roi):
        # Get image size and search window size
        x, y, w, h = roi
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.target_sz = np.array([h, w])
        self.target_sz_real = np.array([h, w])
        self.pos = np.array([y + np.floor(h/2), x + np.floor(w/2)])
        if np.sqrt(h * w) >= 100:  # diagonal size >= threshold
            self.resize = True
            self.pos = np.floor(self.pos / 2)
            self.target_sz = np.floor(self.target_sz / 2)
        if self.resize:
            self.image = cv2.resize(self.image, (self.image.shape[1] // 2, self.image.shape[0] // 2))
        # window size, taking padding into account
        self.window_sz = np.floor(np.multiply(self.target_sz, (1 + self.padding)))
        self.output_sigma = round(round(np.sqrt(self.target_sz[0]*self.target_sz[1]), 4) * self.output_sigma_factor / self.cell_size, 4)
        yf_sz = np.floor(self.window_sz / self.cell_size)
        yf_sz[0] = np.floor(self.window_sz / self.cell_size)[1]
        yf_sz[1] = np.floor(self.window_sz / self.cell_size)[0]
        gauss = gaussian2d_rolled_labels(yf_sz, self.output_sigma)
        self.yf = fft2(gauss)
        #store pre-computed cosine window
        self.cos_window = cos_window([self.yf.shape[1], self.yf.shape[0]])
        # obtain a subwindow for training at newly estimated target position
        patch = self.get_subwindow(self.image, self.pos, self.window_sz)
        feat = self.get_features(patch)
        xf = fftn(feat, axes=(0, 1))
        kf = []
        if self.correlation_type == 'gaussian':
            kf = self.gaussian_correlation(xf, xf)
        alphaf = np.divide(self.yf, (kf + self.lambda_))
        self.model_alphaf = alphaf
        self.model_xf = xf

    def update(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.resize:
            self.image = cv2.resize(self.image, (self.image.shape[1] // 2, self.image.shape[0] // 2))

        patch = self.get_subwindow(self.image, self.pos, self.window_sz)
        zf = fftn(self.get_features(patch), axes=(0, 1))

        if self.correlation_type == 'gaussian':
            kzf = self.gaussian_correlation(zf, self.model_xf)

        response = real(ifftn(self.model_alphaf * kzf, axes=(0, 1)))  # equation for fast detection
        # Find indices and values of nonzero elements curr = np.unravel_index(np.argmax(gi, axis=None), gi.shape)
        delta = np.unravel_index(np.argmax(response, axis=None), response.shape)
        vert_delta, horiz_delta = delta[0], delta[1]
        if vert_delta > np.size(zf, 0) / 2:  # wrap around to negative half-space of vertical axis
            vert_delta = vert_delta - np.size(zf, 0)
        if horiz_delta > np.size(zf, 1) / 2:  # same for horizontal axis
            horiz_delta = horiz_delta - np.size(zf, 1)
        self.pos = self.pos + self.cell_size * np.array([vert_delta, horiz_delta])

        # obtain a subwindow for training at newly estimated target position
        patch = self.get_subwindow(self.image, self.pos, self.window_sz)
        feat = self.get_features(patch)
        xf = fftn(feat, axes=(0, 1))

        # Kernel Ridge Regression, calculate alphas (in Fourier domain)
        if self.correlation_type == 'gaussian':
            kf = self.gaussian_correlation(xf, xf)

        alphaf = np.divide(self.yf, (kf + self.lambda_))
        # subsequent frames, interpolate model
        self.model_alphaf = (1 - self.interp_factor) * self.model_alphaf + self.interp_factor * alphaf
        self.model_xf = (1 - self.interp_factor) * self.model_xf + self.interp_factor * xf

        if self.resize:
            pos_real = np.multiply(self.pos, 2)
        else:
            pos_real = self.pos

        box = [pos_real[1] - self.target_sz_real[1] / 2,
               pos_real[0] - self.target_sz_real[0] / 2,
               self.target_sz_real[1],
               self.target_sz_real[0]]

        return box[0], box[1], box[2], box[3]

    def get_subwindow(self, im, pos, sz):
        _p1 = np.array(range(0, int(sz[0]))).reshape([1, int(sz[0])])
        _p2 = np.array(range(0, int(sz[1]))).reshape([1, int(sz[1])])
        ys = np.floor(pos[0]) + _p1 - np.floor(sz[0]/2)
        xs = np.floor(pos[1]) + _p2 - np.floor(sz[1]/2)

        # Check for out-of-bounds coordinates, and set them to the values at the borders
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs > np.size(im, 1) - 1] = np.size(im, 1) - 1
        ys[ys > np.size(im, 0) - 1] = np.size(im, 0) - 1
        xs = xs.astype(int)
        ys = ys.astype(int)
        # extract image
        out1 = im[list(ys[0, :]), :, :]
        out = out1[:, list(xs[0, :]), :]

        return out

    def get_features(self, im):
        if self.feature == 'hog':
            # HOG features, from Piotr's Toolbox
            x = np.double(self.get_fhog(im))
            return x * self.cos_window[:, :, None]
        if self.feature == 'gray':
            x = np.double(im) / 255
            x = x - np.mean(x)
            return x * self.cos_window[:, :, None]
        if self.feature == 'deep':
            x = self.get_deep_feature(im)
            x = x / np.max(x)
            return x * self.cos_window[:, :, None]

    def get_fhog(self, im_patch):
        H = get_hog(im_patch/255)
        return H

    def gaussian_correlation(self, xf, yf):
        N = xf.shape[0] * xf.shape[1]
        xff = xf.reshape([xf.shape[0] * xf.shape[1] * xf.shape[2], 1], order='F')
        xff_T = xff.conj().T
        yff = yf.reshape([yf.shape[0] * yf.shape[1] * yf.shape[2], 1], order='F')
        yff_T = yff.conj().T
        xx = np.dot(xff_T, xff).real / N  # squared norm of x
        yy = np.dot(yff_T, yff).real / N  # squared norm of y
        # cross-correlation term in Fourier domain
        xyf = xf * conj(yf)
        ixyf = ifftn(xyf, axes=(0, 1))
        rxyf = real(ixyf)
        xy = np.sum(rxyf, 2)  # to spatial domain

        # calculate gaussian response for all positions, then go back to the Fourier domain
        sz = xf.shape[0] * xf.shape[1] * xf.shape[2]
        mltp = (xx + yy - 2 * xy) / sz
        crpm = -1 / (self.sigma * self.sigma)
        expe = crpm * np.maximum(0, mltp)
        expx = np.exp(expe)
        kf = fftn(expx, axes=(0, 1))

        return kf

    def get_deep_feature(self, im):
        # Preprocessing
        from numpy import expand_dims
        #img = im.astype('float32')  # note: [0, 255] range
        img = im  # note: [0, 255] range
        img = cv2.resize(img, (224, 224))

        img = expand_dims(img, axis=0)
        feature_maps = vgg_model.predict(img)
        f_map = feature_maps[3][0][:][:][:]
        feature_map_n = cv2.resize(f_map, (self.cos_window.shape[1], self.cos_window.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

        return feature_map_n



