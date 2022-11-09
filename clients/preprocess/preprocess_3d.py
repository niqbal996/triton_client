from .base_preprocess import Preprocess
import yaml
from easydict import EasyDict
import numpy as np
try:
    from pcdet.datasets import processor
except Exception as e:
    print('[ERROR] {}'.format(e))

class PointpillarPreprocess(Preprocess):

    def __init__(self):
        with open('./data/kitti_dataset.yaml', 'r') as f:
            self.dataset_cfg = EasyDict(yaml.safe_load(f))
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = processor.point_feature_encoder.PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = processor.data_processor.DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, 
            point_cloud_range=self.point_cloud_range,
            training=False, 
            num_point_features=self.point_feature_encoder.num_point_features
        )
        
    def preprocess(self):
        pass

    def filter_pc(self, pointcloud_array):
        '''
        pointcloud_array: input numpy array of shape [N,3] or [N,4] with/without reflectivity
        return: Dictionary with following keys:
        data_dict['points'] = (N, 4)        total number of points in the scan (x, y, z, r)
        data_dict['voxels'] = (Number of filled voxels, MAX_POINTS_PER_VOXEL, 3)
        data_dict['voxel_coords'] = (Number of filled voxels, 3)
        data_dict['voxel_num_points'] = (Number of filled voxels,)
        '''
        data_dict = {
            'points': pointcloud_array,
            'frame_id': 1,
        }

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        return data_dict