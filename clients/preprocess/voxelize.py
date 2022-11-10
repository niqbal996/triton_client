from .base_preprocess import Preprocess
import yaml
from easydict import EasyDict
import numpy as np
try:
    from det3d.torchie import Config
    from det3d.core.input.voxel_generator import VoxelGenerator
except Exception as e:
    print('[ERROR] {}'.format(e))

class det3DPreprocess(Preprocess):

    def __init__(self, cfg='./data/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py'):
        self.cfg = Config.fromfile(cfg)
        self.range = self.cfg.voxel_generator.range
        self.voxel_size = self.cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = self.cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = self.cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
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
        pad = np.zeros((pointcloud_array.shape[0], 1))
        pointcloud_array = np.concatenate((pointcloud_array, pad), axis=1)
        self.points = self.voxel_generator.generate(pointcloud_array)
        #convert to list instead of tuple
        self.points = list(self.points)
        #convert data type of voxels from float64 to float32
        self.points[0] = self.points[0].astype(np.float32)
        # add batch dimension at index 0 to coordinates
        pad = np.zeros((self.points[1].shape[0], 1), dtype=np.int32)
        self.points[1] = np.concatenate((pad, self.points[1]), axis=1)

        return self.points