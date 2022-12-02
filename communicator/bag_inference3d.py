import sys
import os
import time
import rospy
from copy import copy
from pyquaternion import Quaternion
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
try:
    from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
except:
    from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, BoundingBox3D
from geometry_msgs.msg import PoseWithCovariance
import rosbag
from cv_bridge import CvBridge
try:
    import ros_numpy
except Exception as E:
    print('[WARNING] {}'.format(E))
try:
    # import open3d
    from clients.postprocess import visualize_open3d as V
except Exception as E:
    print('[WARNING] {}'.format(E))

# try:
#     from clients.postprocess import visualize_mayavi  as V
# except Exception as E:
#     print('[WARNING] {}'.format(E))

import numpy as np

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc
from .channel import grpc_channel
from .base_inference import BaseInference
# from utils import image_util


class RosInference3D(BaseInference):

    """
    A RosInference to support ROS input and provide input to channel for inference.
    """

    def __init__(self, channel, client, bagfile):
        '''
            channel: channel of type communicator.channel
            client: client of type clients

        '''

        super().__init__(channel, client)

        self.image = None
        self.br = CvBridge()

        self._register_inference() # register inference based on type of client
        self.client_postprocess = client.get_postprocess() # get postprocess of client
        self.client_preprocess = client.get_preprocess()
        self.class_names = self.client_postprocess.load_class_names()
        self.in_bag = rosbag.Bag(bagfile, "r")
        self.out_bag = rosbag.Bag('{}_output.bag'.format(os.path.basename(bagfile)), 'w')
        self.count = 0
        self.box_topic = rospy.Publisher(self.channel.params['pub_topic'], BoundingBoxArray, queue_size=1)

    def _register_inference(self):
        """
        register inference
        """
        # for GRPC channel
        if type(self.channel) == grpc_channel.GRPCChannel:
            self._set_grpc_channel_members()
        else:
            pass

    def _set_grpc_channel_members(self):
        """
            Set properties for grpc channel, queried from the server.
        """
        # collect meta data of model and configuration
        meta_data = self.channel.get_metadata()
        self.input_meta, self.output_meta = self.client.parse_model(
            meta_data["metadata_response"], meta_data["config_response"].config)
        # self.channel.input, self.channel.output, self.input_types = 1
        self.channel.input = [input['name'] for input in self.input_meta]
        self.channel.output = [output['name'] for output in self.output_meta]

        self.dtypes = {'FP32' : 'float32',
                       'FP64' : 'float64',
                       'INT16': 'int16',
                       'INT32': 'int32'
                       # TODO add more types in future
                       }
        self.outputs = {}
        for output,i  in zip(self.output_meta, range(len(self.output_meta))):
            self.outputs['output_{}'.format(i)] = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.outputs['output_{}'.format(i)].name = output['name']
            self.channel.request.outputs.extend([self.outputs['output_{}'.format(i)]])

        self.inputs = {}
        for input, i in zip(self.input_meta, range(len(self.input_meta))):
            self.inputs['input_{}'.format(i)] = service_pb2.ModelInferRequest().InferInputTensor()
            self.inputs['input_{}'.format(i)].name = input['name']
            self.inputs['input_{}'.format(i)].datatype = input['dtype']
            if -1 in input['shape']:
                input['shape'][0] = 10000           # tmp
            self.inputs['input_{}'.format(i)].shape.extend(input['shape'])
            self.channel.request.inputs.extend([self.inputs['input_{}'.format(i)]])

    def yaw2quaternion(self, yaw: float) -> Quaternion:
        return Quaternion(axis=[0,0,1], radians=yaw)

    def start_inference(self):
        offset = 1.5
        for topic, msg, t in self.in_bag.read_messages(topics=[self.channel.params['sub_topic'],
                                                            # '/tf',
                                                            # '/tf_static',
                                                            # '/ai_test_field/sensors/zed2i/zed_node/left_raw/image_raw_color/compressed',
                                                            ]):
            self.count += 1
            t1 = time.time()
            # if topic == '/tf_static':
            #     self.out_bag.write(topic, msg)
            # elif topic == '/ai_test_field/sensors/zed2i/zed_node/left_raw/image_raw_color/compressed':
            #     self.out_bag.write(topic, msg)
            if topic == self.channel.params['sub_topic']:
                self.pc = np.array(list(point_cloud2.read_points(msg, field_names = ("x", "y", "z", "intensity"), skip_nans=True)))
                self.pc[:, 3] = self.pc[:, 3] / np.max(self.pc[:, 3])
                self.pc[:, 2] = self.pc[:, 2] + offset
                tmp0 = self.pc[:, 0]
                tmp1 = self.pc[:, 1]
                self.pc[:,0] = tmp1
                self.pc[:,1] = tmp0
                self.pc = self.client_preprocess.filter_pc(self.pc)
                # the number of voxels changes every sample
                num_voxels = self.pc[0].shape[0]
                self.channel.request.ClearField("raw_input_contents")  # Flush the previous image content
                for key, idx in zip(self.inputs, range(len(self.inputs))):
                    tmp_shape = self.inputs[key].shape
                    self.inputs[key].ClearField("shape")
                    tmp_shape[0] = num_voxels
                    self.channel.request.inputs[idx].ClearField("shape")
                    self.channel.request.inputs[idx].shape.extend(tmp_shape)
                    self.inputs[key].shape.extend(tmp_shape)
                
                # Make sure the data types are correct for each input before sending them as bytes, this causes wrong array values on the server 
                assert self.pc[0].dtype.name == self.dtypes[self.inputs['input_0'].datatype]
                assert self.pc[1].dtype.name == self.dtypes[self.inputs['input_1'].datatype]
                assert self.pc[2].dtype.name == self.dtypes[self.inputs['input_2'].datatype]
                self.channel.request.raw_input_contents.extend([self.pc[0].tobytes(),
                                                                self.pc[1].tobytes(),
                                                                self.pc[2].tobytes(),
                                                                ])
                self.channel.response = self.channel.do_inference() # perform the channel Inference
                self.output = self.client_postprocess.extract_boxes(self.channel.response)
                box_array = self.output['boxes3d_lidar']
                scores = self.output['scores']
                labels = self.output['labels']
                # indices = np.where(scores > 0.5)[0].tolist()
                indices = np.where((labels == 9) & (scores > 0.2))[0].tolist()
                detection_array = BoundingBoxArray()
                if scores.size != 0:
                    for i in indices:
                        bbox = BoundingBox()
                        bbox.header.frame_id = msg.header.frame_id
                        bbox.header.stamp = rospy.Time.now()
                        q = self.yaw2quaternion(float(box_array[i][8]))
                        bbox.pose.orientation.x = q[1]
                        bbox.pose.orientation.y = q[2]
                        bbox.pose.orientation.z = q[3]
                        bbox.pose.orientation.w = q[0]           
                        bbox.pose.position.x = float(box_array[i][0])
                        bbox.pose.position.y = float(box_array[i][1])
                        bbox.pose.position.z = float(box_array[i][2]) - offset
                        bbox.dimensions.x = float(box_array[i][4])
                        bbox.dimensions.y = float(box_array[i][3])
                        bbox.dimensions.z = float(box_array[i][5])
                        bbox.value = scores[i]
                        bbox.label = int(labels[i])
                        detection_array.boxes.append(bbox)
                detection_array.header.stamp = msg.header.stamp
                detection_array.header.frame_id = msg.header.frame_id
                t2 = time.time()
                # print('[INFO] Time taken {} s'.format(t2 -t1))
                print('[INFO] Frame ID {} '.format(msg.header.seq))
                self.out_bag.write(topic, msg)
                self.out_bag.write(self.channel.params['pub_topic'], detection_array)
                detection_array = []