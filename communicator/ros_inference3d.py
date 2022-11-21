import sys
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

    def __init__(self, channel, client, jsk=True):
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
        self.jsk = jsk

    def _register_inference(self):
        """
        register inference
        """
        # for GRPC channel
        if type(self.channel) == grpc_channel.GRPCChannel:
            self._set_grpc_channel_members()
        else:
            pass

        # self.detection = rospy.Publisher(self.channel.params['pub_topic'], Image, queue_size=10)

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

    def start_inference(self):
        rospy.Subscriber(self.channel.params['sub_topic'], PointCloud2, self._pc_callback, queue_size=1)
        if self.jsk:
            self.publisher = rospy.Publisher(self.channel.params['pub_topic'], BoundingBoxArray, queue_size=1)
        else:
            self.publisher = rospy.Publisher(self.channel.params['pub_topic'], Detection3DArray, queue_size=1)
        rospy.spin()

    def yaw2quaternion(self, yaw: float) -> Quaternion:
        return Quaternion(axis=[0,0,1], radians=yaw)

    def _pc_callback(self, msg):
        t1 = time.time()
        # self.pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        # TODO what is the 4th attribute of the point clouds from KITTI and what is their data range
        self.pc = np.array(list(point_cloud2.read_points(msg, field_names = ("x", "y", "z", "reflectivity"), skip_nans=True)))
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

        # self.input0.ClearField("shape")
        # self.input1.ClearField("shape")
        # self.input2.ClearField("shape")
        # self.input0.shape.extend([num_voxels, 32, 4])
        # self.input1.shape.extend([num_voxels, 4])
        # self.input2.shape.extend([num_voxels])
        # self.channel.request.ClearField("inputs")
        
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
        # https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md
        # [12, 13, 14, 18]:
        # indices1 = np.where(scores > 0.5)[0].tolist()
        indices = np.where(labels == 9)[0].tolist()
        # print(np.unique(labels))
        if self.jsk:
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
                    bbox.pose.position.z = float(box_array[i][2])
                    bbox.dimensions.x = float(box_array[i][4])
                    bbox.dimensions.y = float(box_array[i][3])
                    bbox.dimensions.z = float(box_array[i][5])
                    bbox.value = scores[i]
                    bbox.label = int(labels[i])
                    detection_array.boxes.append(bbox)
        else:
            detection_array = Detection3DArray()
            # for idx in indices:
            for idx in range(len(box_array)):
                detection = Detection3D()
                bbox3D = BoundingBox3D()
                object_hypothesis = ObjectHypothesisWithPose()
                bbox3D.center.position.x = box_array[idx, 0]
                bbox3D.center.position.y = box_array[idx, 1]
                bbox3D.center.position.z = box_array[idx, 2]
                q = self.yaw2quaternion(float(box_array[idx, 8]))
                bbox3D.center.orientation.w = q[0]
                bbox3D.center.orientation.x = q[1]
                bbox3D.center.orientation.y = q[2]
                bbox3D.center.orientation.z = q[3]
                bbox3D.size.x = box_array[idx, 4]
                bbox3D.size.y = box_array[idx, 3]
                bbox3D.size.z = box_array[idx, 5] 
                object_hypothesis.score = copy(scores[idx])
                object_hypothesis.id = copy(labels[idx])
                detection.bbox = bbox3D
                bbox3D = None   #Flush 
                detection.header = msg.header
                detection.results.append(object_hypothesis)
                object_hypothesis = None # Flush
                detection_array.detections.append(detection)
                detection = None    #Flush
            # detection_array.header.stamp = rospy.Time.now()
        detection_array.header.stamp = msg.header.stamp
        detection_array.header.frame_id = msg.header.frame_id
        t2 = time.time()
        print('[INFO] Time taken {} s'.format(t2 -t1))
        self.publisher.publish(detection_array)
        detection_array = []
