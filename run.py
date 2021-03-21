import os
from os.path import join, dirname, realpath
from kernelized_correlation_filter import KernelizedCorrelationFilter
from utils import get_ground_truthes, get_img_list


def run():
    data_dir = 'dataset'
    data_names = sorted(os.listdir(data_dir))

    try:
        for data_name in data_names:
            tracker = KernelizedCorrelationFilter(correlation_type='gaussian', feature='deep')

            data_path = join(data_dir, data_name)
            gts = get_ground_truthes(data_path)

            img_dir = os.path.join(data_path, 'img')
            frame_list = get_img_list(img_dir)
            frame_list.sort()
            poses = tracker.start(init_gt=gts[0], show=True, frame_list=frame_list)
            print(poses)

    except Exception as e:
        print(e)


'''
for HOG

tracker = KernelizedCorrelationFilter(correlation_type='gaussian', feature='hog', vgg_path=vgg_path)

for VGG-19

tracker = KernelizedCorrelationFilter(correlation_type='gaussian', feature='deep', vgg_path=vgg_path)

'''

if __name__ == '__main__':
    run()
