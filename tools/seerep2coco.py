import json
import time
import numpy as np
import copy
import itertools
from . import mask as maskUtils
import os
from collections import defaultdict
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class COCO_SEEREP:
    def __init__(self, seerep_data=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.seerep_data= seerep_data
        self.coco_dict = None
        self.gt = None
        self.preds = None 

    def init_gt(self):
        
    