import numpy as np
import os

def gaussian2d_rolled_labels(sz,sigma):
    w, h = sz
    xs, ys = np.meshgrid(np.arange(w)-w//2, np.arange(h)-h//2)
    dist = (xs**2+ys**2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    labels = np.roll(labels, -int(np.floor(sz[0] / 2)), axis=1)
    labels = np.roll(labels, -int(np.floor(sz[1]/2)), axis=0)

    return labels

def cos_window(sz):
    """
    width, height = sz
    j = np.arange(0, width)
    i = np.arange(0, height)
    J, I = np.meshgrid(j, i)
    cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
    """

    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])

    return cos_window

def get_ground_truthes(img_path):
    gt_path = os.path.join(img_path, 'groundtruth_rect.txt')

    gts = []
    with open(gt_path, 'r') as f:
        while True:
            line = f.readline()
            if line == '':
                gts = np.array(gts, dtype=np.float32)
                return gts
            if ',' in line:
                gt_pos = line.split(',')
            else:
                gt_pos = line.split()
            gt_pos_int = [(float(element)) for element in gt_pos]
            gts.append(gt_pos_int)

def get_img_list(img_dir):
    frame_list = []
    for frame in sorted(os.listdir(img_dir)):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_dir, frame))
    return frame_list
