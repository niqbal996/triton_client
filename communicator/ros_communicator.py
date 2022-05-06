import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

from .base_communicator import BaseCommunicator
from utils import image_util


class RosCommunicator(BaseCommunicator):

    def __init__(self, params, FLAGS, client):
        '''
            grpc_stub: gRPC stub to invoke ModelInfer() in this class
            input_name: Name of the input as described by the onnx converter
            output_name: Model output name as described by the onnx converter
            param: Parameters taken from the yaml file for ros topic names and server URL
            dtype: Data type of the input
            c,h,w: Channel width height

        '''
        super().__init__(params, FLAGS, client)

        self.image = None
        self.br = CvBridge()
        self.class_names = image_util.load_class_names()
        self.input = service_pb2.ModelInferRequest().InferInputTensor()
        self._register_communicator()

    def _register_communicator(self):
        """

        """
        meta_data = self.get_metadata()
        self.input.name, output_name, c, h, w, format, self.input.datatype = self.client.parse_model(
            meta_data["metadata_response"], meta_data["config_response"].config)
        self.input_size = [h, w]
        if format == mc.ModelInput.FORMAT_NHWC:
            self.input.shape.extend([h, w, c])
        else:
            self.input.shape.extend([c, h, w])

        self.request = service_pb2.ModelInferRequest()
        self.request.model_name = self.FLAGS.model_name
        self.request.model_version = self.FLAGS.model_version
        # TODO make it dynamic for all models
        if False:
            self.output0 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.output0.name = output_name[0]
            self.output1 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.output1.name = output_name[1]
            self.request.outputs.extend([self.output0, self.output1])
        else:
            self.output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.output.name = output_name[0]
            self.request.outputs.extend([self.output])
        self.detection = rospy.Publisher(self.params['pub_topic'], Image, queue_size=10)

    def start_inference(self):
        rospy.Subscriber(self.params['sub_topic'], Image, self._callback)
        rospy.spin()

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
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.orig_size = cv_image.shape[0:2]
        self.orig_image = cv_image.copy()
        cv_image = cv2.resize(cv_image, (self.input.shape[1], self.input.shape[2]))
        self.image = image_util.image_adjust(cv_image).astype(np.float32)
        if self.image is not None:
            self.request.ClearField("inputs")
            self.request.ClearField("raw_input_contents")  # Flush the previous image contents
            self.request.inputs.extend([self.input])
            self.request.raw_input_contents.extend([self.image.tobytes()])
            # self.request.inputs.in
            self.response = self.get_mode().ModelInfer(self.request)  # Inference
            self.prediction = self.client_postprocess.extract_boxes_yolov5(self.response)

            for object in self.prediction[
                0]:  # predictions array has the order [x1,y1, x2,y2, confidence, confidence, class ID]
                box = np.array(object[0:4], dtype=np.float32)
                box = self._scale_boxes(box, normalized=False)
                if int(object[5]) == 0:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(self.orig_image,
                              pt1=(int(box[0]), int(box[1])),
                              pt2=(int(box[2]), int(box[3])),
                              color=color,
                              thickness=3)
                # cv2.putText(self.orig_image,
                #             '{:.2f} {}'.format(object[-2], self.class_names[int(object[-1])]),
                #             org=(int(box[0]), int(box[1])),
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=0.5,
                #             thickness=2,
                #             color=(0, 255, 0))
            # cv2.imshow('Prediction', cv2.cvtColor(self.orig_image, cv2.COLOR_RGB2BGR))
            # cv2.imshow('Prediction', cv2.cvtColor(self.orig_image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            self.msg_frame = self.br.cv2_to_imgmsg(self.orig_image, encoding="rgb8")
            self.msg_frame.header.stamp = rospy.Time.now()
            self.detection.publish(self.msg_frame)
