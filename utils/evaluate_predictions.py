import rospy
from sensor_msgs.msg import Image
# from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc
from utils.postprocess import extract_boxes_triton, extract_boxes_yolov5, xywh2xyxy, box_iou
from utils.pred2ros_msg import yolo2vision_msg

class EvaluatorNode(object):
    def __init__(self,
                 grpc_stub,
                 input_name,
                 output_name,
                 param,
                 FLAGS,
                 dtype,
                 c, h, w,
                 visualize_gt=False,
                 visualize_pred=False):
        '''
        grpc_stub: gRPC stub to invoke ModelInfer() in this class
        input_name: Name of the input as described by the onnx converter
        output_name: Model output name as described by the onnx converter
        param: Parameters taken from the yaml file for ros topic names and server URL
        dtype: Data type of the input
        c,h,w: Channel width height

        '''
        self.image = None
        self.br = CvBridge()
        self.stub = grpc_stub
        self.FLAGS = FLAGS
        self.param = param
        self.class_names = self.load_class_names()
        self.input = service_pb2.ModelInferRequest().InferInputTensor()
        self.input.name = input_name
        self.input.datatype = dtype
        self.input_size = [h, w]
        self.count  = 0
        self.id_list_preds = []
        self.id_list_gts = []
        self.all_predictions = []
        self.all_groundtruths = []
        self.show_preds = visualize_pred
        self.show_gt = visualize_gt
        if format == mc.ModelInput.FORMAT_NHWC:
            self.input.shape.extend([h, w, c])
        else:
            self.input.shape.extend([c, h, w])

        self.request = service_pb2.ModelInferRequest()
        self.request.model_name = FLAGS.model_name
        self.request.model_version = FLAGS.model_version
        # TODO make it dynamic for all models
        # if False:
            # self.output0 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            # self.output0.name = output_name[0]
            # self.output1 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            # self.output1.name = output_name[1]
            # self.request.outputs.extend([self.output0, self.output1])
        # else:
        self.output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        self.output.name = output_name[0]
        self.request.outputs.extend([self.output])

    def start_evaluation(self):
        self.inference_topic = rospy.Subscriber(self.param['sub_topic'], Image, self.image_callback)
        self.gt_topic = rospy.Subscriber(self.param['gt_topic'], Detection2DArray, self.gt_callback)
        rospy.spin()
        if self.show_gt:
            self.visualize_gt()
        if self.show_preds:
            self.visualize_pred()
        self.calculate_metrics()

    def calculate_metrics(self):
        import torch
        iou = box_iou(torch.tensor(self.all_groundtruths[0]['ground_truths'][:, :4]),
                    torch.tensor(self.all_predictions[0]['prediction'][:, :4]))
        iouv = [0.0, 0.1, 0.2, 0.3, 0.5]
        x = torch.where((iou >= iouv[0]) & (self.all_groundtruths[0]['ground_truths'][:, 5] == self.all_predictions[0]['prediction'][:, 5]))  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                1).cpu().numpy()  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        return correct

    def scale_boxes(self, box, normalized=False):
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

    def image_callback(self, msg):
        # rospy.loginfo('Image received...')
        # rospy.loginfo('Message count {}'.format(self.count))
        self.count += 1
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.orig_size = cv_image.shape[0:2]
        self.orig_image = cv_image.copy()
        cv_image = cv2.resize(cv_image, (self.input.shape[1], self.input.shape[2]))
        self.image = self.image_adjust(cv_image).astype(np.float32)
        if self.image is not None:
            self.request.ClearField("inputs")
            self.request.ClearField("raw_input_contents")   # Flush the previous image contents
            self.request.inputs.extend([self.input])
            self.request.raw_input_contents.extend([self.image.tobytes()])
            self.response = self.stub.ModelInfer(self.request)  # Inference
            self.prediction = extract_boxes_yolov5(self.response)
            if msg.header.seq not in self.id_list_preds:
                self.id_list_preds.append(msg.header.seq)
                self.all_predictions.append({
                    'header': msg.header.seq,
                    'prediction': self.prediction[0].cpu().numpy(),
                })
            elif msg.header.seq in self.id_list_preds:
                # this ros bag is done now stop spinning it.
                rospy.loginfo('Closing the subscriber topic: {}'.format(self.param['sub_topic']))
                self.inference_topic.unregister()
                rospy.signal_shutdown('Closing rospy for postprocessing!')

    def gt_callback(self, msg):
        # rospy.loginfo('Detection received...')
        # rospy.loginfo('Message count {}'.format(self.count))
        # self.count +=1
        gt = {
            'header': msg.detections[0].source_img.header.seq,
            'ground_truths': None
        }
        if msg.detections[0].source_img.header.seq not in self.id_list_gts:
            self.id_list_gts.append(msg.header.seq)
            gts = np.zeros((len(msg.detections), 6), dtype=float)

            for detection, idx in zip(msg.detections, range(len(msg.detections))):
                gts[idx, :] = [detection.bbox.center.x,
                               detection.bbox.center.y,
                               detection.bbox.size_x,
                               detection.bbox.size_y,
                               detection.results[0].score,
                               detection.results[0].id]
            # convert to x1,y1 and x2,y2 format
            gts = xywh2xyxy(gts)
            gt['ground_truths'] = gts
            self.all_groundtruths.append(gt)
        elif msg.header.seq in self.id_list_gts:
            rospy.loginfo('Closing the subscriber topic: {}'.format(self.param['gt_topic']))
            self.gt_topic.unregister()

    def visualize_gt(self):

        for object in self.prediction[0]:  # [x1,y1, x2,y2, confidence, confidence, class ID]
            box = np.array(object[0:4], dtype=np.float32)
            box = self.scale_boxes(box, normalized=False)
            if int(object[5]) == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(self.orig_image,
                          pt1=(int(box[0]), int(box[1])),
                          pt2=(int(box[2]), int(box[3])),
                          color=color,
                          thickness=3)
            cv2.putText(self.orig_image,
                        '{:.2f} {}'.format(object[-2], self.class_names[int(object[-1])]),
                        org=(int(box[0]), int(box[1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=2,
                        color=(0, 255, 0))
        cv2.imshow('Prediction', cv2.cvtColor(self.orig_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

    def visualize_pred(self):

        for object in self.prediction[0]:  # [x1,y1, x2,y2, confidence, confidence, class ID]
            box = np.array(object[0:4], dtype=np.float32)
            box = self.scale_boxes(box, normalized=False)
            if int(object[5]) == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(self.orig_image,
                          pt1=(int(box[0]), int(box[1])),
                          pt2=(int(box[2]), int(box[3])),
                          color=color,
                          thickness=3)
            cv2.putText(self.orig_image,
                        '{:.2f} {}'.format(object[-2], self.class_names[int(object[-1])]),
                        org=(int(box[0]), int(box[1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=2,
                        color=(0, 255, 0))
        cv2.imshow('Prediction', cv2.cvtColor(self.orig_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

    def load_class_names(self, namesfile='./data/crop.names'):
        class_names = []
        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names

    def image_adjust(self, cv_image):
        '''
        cv_image: input image in RGB order
        return: Normalized image in BCHW dimensions.
        '''
        # pad = np.zeros((16, 1280, 3), dtype=np.uint8)
        # cv_image = np.concatenate((cv_image, pad), axis=0)
        # orig = cv_image.copy()
        cv_image = np.transpose(cv_image, (2, 0, 1)).astype(np.float32)
        cv_image = np.expand_dims(cv_image, axis=0)
        cv_image /= 255.0

        return cv_image

if __name__ == '__main__':
    rospy.init_node("Evaluator", anonymous=True)
    my_node = EvaluatorNode()