import numpy as np
import cv2 as cv


def load_class_names(namesfile='./data/coco.names'):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def image_adjust(cv_image):
    '''
    cv_image: input image in RGB order
    return: Normalized image in BCHW dimensions.
    '''
    # pad = np.zeros((16, 1280, 3), dtype=np.uint8)
    # cv_image = np.concatenate((cv_image, pad), axis=0)
    # orig = cv_image.copy()
    cv_image = np.transpose(cv_image, (2, 0, 1)).astype(np.float32)
    cv_image = np.expand_dims(cv_image, axis=0)
    cv_image /= 255.0

    return cv_image