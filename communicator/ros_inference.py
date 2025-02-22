import rospy
from sensor_msgs.msg import Image, PointCloud2, CompressedImage
from cv_bridge import CvBridge
try:
    import ros_numpy
except ImportError:
    print("[WARNING] ros_numpy was not imported")

import cv2
import numpy as np

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc
from .channel import grpc_channel
from .base_inference import BaseInference
# from utils import image_util


class RosInference(BaseInference):

    """
    A RosInference to support ROS input and provide input to channel for inference.
    """

    def __init__(self, channel, client):
        '''
            channel: channel of type communicator.channel
            client: client of type clients

        '''

        super().__init__(channel, client)
        self.topic_exists = False
        self.image = None
        self.br = CvBridge()
        self.color = (0, 255, 0)
        self.topic_exists = False
        self._register_inference() # register inference based on type of client
        self.client_postprocess = client.get_postprocess() # get postprocess of client
        self.client_preprocess = client.get_preprocess()
        self.class_names = self.client_postprocess.load_class_names()

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

        # parse the model requirements from client
        self.channel.input.name, output_name, c, h, w, format, self.channel.input.datatype = self.client.parse_model(
            meta_data["metadata_response"], meta_data["config_response"].config)

        self.input_size = [h, w]
        if format == mc.ModelInput.FORMAT_NHWC:
            self.channel.input.shape.extend([h, w, c])
        else:
            self.channel.input.shape.extend([c, h, w])

        if len(output_name) > 1:  # Models with multiple outputs Boxes, Classes and scores
            self.output0 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # boxes
            self.output0.name = output_name[0]
            self.output1 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # class_IDs
            self.output1.name = output_name[1]
            self.output2 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # scores
            self.output2.name = output_name[2]
            self.output3 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # image dims
            self.output3.name = output_name[3]

            self.channel.request.outputs.extend([self.output0,
                                                 self.output1,
                                                 self.output2,
                                                 self.output3])
        else:
            self.output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.output.name = output_name[0]
            self.channel.request.outputs.extend([self.output])
        # self.channel.output.name = output_name[0]
        # self.channel.request.outputs.extend([self.channel.output])

    def start_inference(self):
        rospy.loginfo('Listening to Image topic: {}'.format(self.channel.params['sub_topic']))
        rospy.Subscriber(self.channel.params['sub_topic'], CompressedImage, self._callback)
        self.detection = rospy.Publisher(self.channel.params['pub_topic'], Image, queue_size=10)
        # rospy.Subscriber(self.channel.params['sub_topic'], PointCloud2, self._pc_callback)
        rospy.spin()

    def _check_topic(self, msg):
        rospy.logerr('Ros topic found. Proceeding with inference . . .')
    def _scale_boxes(self, box, normalized=False):
        '''
        box: Bounding box generated for the image size (e.g. 512 x 512) expected by the model at triton server
        return: Scaled bounding box according to the input image from the ros topic.
        '''
        if normalized:
            # TODO make it dynamic with mc.Modelshape according to CHW or HWC
            xtl, xbr = box[0] * self.orig_size[1], box[2] * self.orig_size[1]
            ytl, ybr = box[1] * self.orig_size[0], box[3] * self.orig_size[0]
        else:
            xtl, xbr = box[0] * (self.orig_size[1] / self.input_size[0]), \
                       box[2] * (self.orig_size[1] / self.input_size[0])
            ytl, ybr = box[1] * self.orig_size[0] / self.input_size[1], \
                       box[3] * self.orig_size[0] / self.input_size[1]

        return [xtl, ytl, xbr, ybr]

    def _callback(self, msg):
        # rospy.loginfo('Image received...')
        if 'compressed' in msg.format:
            # cv_image = np.fromstring(msg.data, np.uint8)
            # cv_image = cv2.imdecode(cv_image, cv2.IMREAD_COLOR)
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            str_msg = msg.data
            buf = np.ndarray(shape=(1, len(str_msg)), 
                             dtype=np.uint8, 
                             buffer=msg.data)
            cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)   # BGR8
            # cv2.imshow('image', cv_image)
            # cv2.waitKey(1)
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv_image = cv_image[...,::-1]
        else:
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        if cv_image is not None and not self.topic_exists:
            rospy.loginfo("Topic received. Proceeding with Inference . . .")
            rospy.loginfo("Publishing detections under {}".format(self.channel.params['pub_topic']))
            self.topic_exists = True
        self.orig_size = cv_image.shape[0:2]
        self.orig_image = cv_image.copy()
        cv_image = cv2.resize(cv_image, (self.channel.input.shape[1], self.channel.input.shape[2]))
        self.image = self.client_preprocess.image_adjust(cv_image)
        if self.image is not None:
            self.channel.request.ClearField("inputs")
            self.channel.request.ClearField("raw_input_contents")  # Flush the previous image contents
            self.channel.request.inputs.extend([self.channel.input])
            self.channel.request.raw_input_contents.extend([self.image.tobytes()])
            self.channel.response = self.channel.do_inference() # perform the channel Inference
            self.prediction = self.client_postprocess.extract_boxes(self.channel.response, conf_thres=0.3)
            if not isinstance(self.prediction, AssertionError):
                for object in self.prediction[0]:  # predictions array has the order [x1,y1, x2,y2, confidence,
                                                                                    # confidence, class ID]
                    box = np.array(object[0:4], dtype=np.float32)
                    box = self._scale_boxes(box, normalized=False)
                    # if int(object[5]) == 0:
                    #     color = (0, 255, 0)
                    # else:
                    #     color = (255, 0, 0)
                    cv2.rectangle(self.orig_image,
                                pt1=(int(box[0]), int(box[1])),
                                pt2=(int(box[2]), int(box[3])),
                                color=self.color,
                                thickness=2)
                    cv2.putText(self.orig_image,
                                '{:.2f} {}'.format(object[-2], self.class_names[int(object[-1])]),
                                org=(int(box[0]), int(box[1] - 10 )),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=2,
                                color=self.color)
            # cv2.imshow('Prediction', cv2.cvtColor(self.orig_image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)

            self.msg_frame = self.br.cv2_to_imgmsg(self.orig_image, encoding="rgb8")
            self.msg_frame.header = msg.header 
            self.detection.publish(self.msg_frame)


    # def _pc_callback(self, msg):
    #     self.pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
    #     self.pc = self.client_preprocess.filter_pc(self.pc)
    #     self.channel.request.ClearField("inputs")
    #     self.channel.request.ClearField("raw_input_contents")  # Flush the previous image contents
    #     # TODO these are dummy requests, should implement the model on the server side first.
    #     self.channel.request.inputs.extend([self.channel.input])
    #     self.channel.request.raw_input_contents.extend([self.pc.tobytes()])
    #     self.channel.response = self.channel.do_inference() # perform the channel Inference
