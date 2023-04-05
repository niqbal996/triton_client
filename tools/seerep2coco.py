import json
import time
import numpy as np
import copy
import itertools
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
        self.anns =  {'weeds':0, 
                       'maize':1}
        self.ground_truth = self.init_gt()
        self.predictions = self.init_pred_result()


    def init_gt(self):
        tic = time.time()
        cocoGt=COCO(annotation_file=None)
        self.cocoDataset = dict()
        self.cocoDataset['images'] = []
        self.cocoDataset['licenses'] = []
        self.cocoDataset['annotations'] = []
        self.cocoDataset['categories'] = []
        # initialize  images array, maybe we dont need this for evaluation. 
        for item in self.seerep_data:
            tmp = {}
            tmp['license'] = 1 # dummy license
            tmp['filename'] = 'dontknow'
            tmp['height'] = item['image'].shape[0]
            tmp['width'] = item['image'].shape[1]
            tmp['id'] = item['uuid']
            self.cocoDataset['images'].append(tmp)

            for annotation in item['boxes']:
                # put the annotations into coco format
                tmp = {}
                tmp['segmentation'] = []
                tmp['area'] = annotation[2] * annotation[3]
                tmp['iscrowd'] = 0
                tmp['bbox'] = [annotation[0], annotation[1], annotation[2], annotation[3]]
                tmp['image_id'] = item['uuid']
                tmp['category_id'] = annotation[4]
                tmp['id'] = 1 # this id corresponds to ID of each label, leaving it as 1 for now TODO
                self.cocoDataset['annotations'].append(tmp)
        for cat in self.anns:
            tmp = {}
            tmp['supercategory'] = 'crops'
            tmp['id'] = self.anns[cat]
            tmp['name'] = cat
            self.cocoDataset['categories'].append(tmp)
        cocoGt.dataset = self.cocoDataset
        cocoGt.createIndex()
        return cocoGt

    def init_pred_result(self):
        tic = time.time()
        self.res = COCO()
        self.res.dataset['images'] = self.cocoDataset['images']
        preds = []
        if len(self.cocoDataset['annotations'][0]['bbox']) == 4:
            self.res.dataset['categories'] = copy.deepcopy(self.cocoDataset['categories'])
            for item in self.seerep_data:
                for annotation in item['boxes']:
                    # put the annotations into coco format
                    tmp = {}
                    x1, x2, y1, y2 = [annotation[0], annotation[0]+annotation[2], annotation[1], annotation[1]+annotation[3]]
                    if not 'segmentation' in item:
                        tmp['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                    tmp['area'] = annotation[2] * annotation[3]
                    tmp['iscrowd'] = 0
                    tmp['bbox'] = [annotation[0], annotation[1], annotation[2], annotation[3]]
                    tmp['image_id'] = item['uuid']
                    tmp['category_id'] = annotation[4]
                    tmp['score'] = annotation[5]
                    tmp['id'] = 1 # this id corresponds to ID of each label, leaving it as 1 for now TODO
                    preds.append(tmp)
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))
        self.res.dataset['annotations'] = preds
        self.res.createIndex()
        return self.res

    