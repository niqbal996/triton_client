import numpy as np
from pcdet.datasets import DatasetTemplate


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
        """
        Args:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=None, logger=None)
        # data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        # data_file_list.sort()
        self.sample_file_list = [1, 2, 3]

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        # normalize intensity

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict