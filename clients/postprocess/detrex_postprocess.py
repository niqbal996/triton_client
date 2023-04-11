from .base_postprocess import Postprocess
import numpy as np
import sys
# import struct
# import os
# import math
# import time
# import torch
# import torchvision

class Detrexpostprocess(Postprocess):
    def __init__(self):
        pass

    def postprocess(self):
        pass

    def load_class_names(self, dataset='COCO'):
        if dataset=='COCO':
            namesfile = './data/coco.names'
        elif dataset=='CROP':
            namesfile='./data/crop.names'
        else:
            print('[ERROR] No valid dataset was provided. Exiting!')
            sys.exit(0)
        class_names = []
        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names

    def extract_boxes(self, prediction):
        """Runs Non-Maximum Suppression (NMS) on inference results

            Returns:
                 list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            """
        boxes = self.deserialize_bytes_float(prediction.raw_output_contents[0])
        boxes = np.reshape(boxes, prediction.outputs[0].shape)
        input_shape = self.deserialize_bytes_int(prediction.raw_output_contents[1])
        scores = self.deserialize_bytes_float(prediction.raw_output_contents[2])
        scores = np.reshape(scores, prediction.outputs[2].shape)
        class_ids = self.deserialize_bytes_int(prediction.raw_output_contents[3])
        class_ids = np.reshape(class_ids, prediction.outputs[3].shape)
        conf_inds = np.where(scores > 0.50)
        return [boxes[conf_inds], class_ids[conf_inds], scores[conf_inds]]