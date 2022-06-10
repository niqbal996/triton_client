from .base_postprocess import Postprocess
import numpy as np
# import struct
# import os
# import math
# import time
# import torch
# import torchvision

class FCOSpostprocess(Postprocess):
    def __init__(self):
        pass

    def postprocess(self):
        pass

    def load_class_names(self, namesfile='./data/crop.names'):
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
        # TODO class_ids are all zeros? BUG in unpacking or inferencing?
        # class_ids = deserialize_bytes_int(prediction.raw_output_contents[1])
        # class_ids = np.reshape(class_ids, prediction.outputs[1].shape * 2) # TODO DEBUG: Remove after debug *2
        scores = self.deserialize_bytes_float(prediction.raw_output_contents[2])
        scores = np.reshape(scores, prediction.outputs[2].shape)
        class_ids = [0] * len(scores) # TODO remove after debug! dummy values for class IDs remove
        # prediction = boxes

        return boxes, class_ids, scores