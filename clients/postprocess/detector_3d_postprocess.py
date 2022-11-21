from .base_postprocess import Postprocess
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
# try:
#     from mmcv.ops import nms, nms_rotated
# except ImportError as e:
#     print("[ERROR] {}".format(e))
class PointPillarPostprocess(Postprocess):
    def __init__(self):
        # self.use_sigmoid_cls = True
        # self.use_rotate_nms = True
        # self.feat_map_size = torch.Size([248, 216])
        # self.rotations = [0, 1.5707963]
        # self.box_code_size = 7          # TODO change 7  to dynamic box_code_size
        # self.anchors = self.single_level_grid_anchors(featmap_size=self.feat_map_size,
        #                                               scale=1)        
        pass
    def postprocess(self):
        pass

    def load_class_names(self, namesfile='./data/nuScenes.names'):
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
        self.boxes = self.deserialize_bytes_float(prediction.raw_output_contents[0])
        self.boxes = np.reshape(self.boxes, prediction.outputs[0].shape)
        self.scores = self.deserialize_bytes_float(prediction.raw_output_contents[1])
        self.scores = np.reshape(self.scores, prediction.outputs[1].shape)
        self.labels = self.deserialize_bytes_int(prediction.raw_output_contents[2])
        self.labels = np.reshape(self.labels, prediction.outputs[2].shape)

        self.output = {'boxes3d_lidar': self.boxes,
                       'scores': self.scores,
                       'labels': self.labels
                        }
        self.output = self.remove_low_score_nu(self.output, 0.45)

        return self.output

    # Source https://github.com/CarkusL/CenterPoint/
    def get_annotations_indices(self, types, thresh, label_preds, scores):
        indexs = []
        annotation_indices = []
        for i in range(label_preds.shape[0]):
            if label_preds[i] == types:
                indexs.append(i)
        for index in indexs:
            if scores[index] >= thresh:
                annotation_indices.append(index)
        return annotation_indices  

    # def generate_jsk_messages(self):
    #     arr_bbox = BoundingBoxArray()
    #     for i in range(10):
    #             bbox = BoundingBox()
    #             bbox.header.frame_id = msg.header.frame_id
    #             bbox.header.stamp = rospy.Time.now()
    #             q = yaw2quaternion(float(dt_box_lidar[i][8]))
    #             bbox.pose.orientation.x = q[1]
    #             bbox.pose.orientation.y = q[2]
    #             bbox.pose.orientation.z = q[3]
    #             bbox.pose.orientation.w = q[0]           
    #             bbox.pose.position.x = float(dt_box_lidar[i][0])
    #             bbox.pose.position.y = float(dt_box_lidar[i][1])
    #             bbox.pose.position.z = float(dt_box_lidar[i][2])
    #             bbox.dimensions.x = float(dt_box_lidar[i][4])
    #             bbox.dimensions.y = float(dt_box_lidar[i][3])
    #             bbox.dimensions.z = float(dt_box_lidar[i][5])
    #             bbox.value = scores[i]
    #             bbox.label = int(types[i])
    #             arr_bbox.boxes.append(bbox)
    #             print("total callback time: ", time.time() - t_t)
    #             arr_bbox.header.frame_id = msg.header.frame_id
    #             arr_bbox.header.stamp = msg.header.stamp
    #             if len(arr_bbox.boxes) != 0:
    #                 pub_arr_bbox.publish(arr_bbox)
    #                 arr_bbox.boxes = []
    #             else:
    #                 arr_bbox.boxes = []
    #                 pub_arr_bbox.publish(arr_bbox)


    # Source https://github.com/CarkusL/CenterPoint/
    def remove_low_score_nu(self, predictions, thresh):
        filtered_annotations = {}
        label_preds_ = predictions["labels"]
        scores_ = predictions["scores"]
        
        car_indices =                  self.get_annotations_indices(0, 0.4, label_preds_, scores_)
        truck_indices =                self.get_annotations_indices(1, 0.4, label_preds_, scores_)
        construction_vehicle_indices = self.get_annotations_indices(2, 0.4, label_preds_, scores_)
        bus_indices =                  self.get_annotations_indices(3, 0.3, label_preds_, scores_)
        trailer_indices =              self.get_annotations_indices(4, 0.4, label_preds_, scores_)
        barrier_indices =              self.get_annotations_indices(5, 0.4, label_preds_, scores_)
        motorcycle_indices =           self.get_annotations_indices(6, 0.15, label_preds_, scores_)
        bicycle_indices =              self.get_annotations_indices(7, 0.15, label_preds_, scores_)
        pedestrain_indices =           self.get_annotations_indices(8, 0.1, label_preds_, scores_)
        traffic_cone_indices =         self.get_annotations_indices(9, 0.1, label_preds_, scores_)
        
        for key in predictions.keys():
            if key == 'metadata':
                continue
            filtered_annotations[key] = (
                predictions[key][car_indices +
                                pedestrain_indices + 
                                bicycle_indices +
                                bus_indices +
                                construction_vehicle_indices +
                                traffic_cone_indices +
                                trailer_indices +
                                barrier_indices +
                                truck_indices
                                ])
        # print("[INFO] Filtered {} out of {} predictions based on {} percent confidence threshold".format(
        #     len(filtered_annotations['scores']), 
        #     len(predictions['scores']), 
        #     int(thresh*100)
        #     ))
        return filtered_annotations