from .base_preprocess import Preprocess
from pcdet.datasets import processor
import numpy as np

class PCDetpreprocess(Preprocess):

    def __init__(self):
        pass

    def preprocess(self):
        pass

    def filter_pc(self, cv_image):
        '''
        cv_image: input image in RGB order
        return: Normalized image in BCHW dimensions.
        '''

        self.data_processor = processor.data_processor.DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        return data_dict