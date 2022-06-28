import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge

import time
import cv2
import numpy as np

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc
from .channel import grpc_channel
from .base_inference import BaseInference
# from utils import image_util

from prometheus_client import start_http_server, Summary, Histogram

class EvaluateInference(BaseInference):

    """
    A RosInference to support ROS input and provide input to channel for inference.
    """

    def __init__(self, channel, client):
        '''
            channel: channel of type communicator.channel
            client: client of type clients

        '''

        super().__init__(channel, client)

        self.image = None
        self.br = CvBridge()
        # self.class_names = self.load_class_names()

        self._register_inference() # register inference based on type of client
        self.client_postprocess = client.get_postprocess() # get postprocess of client
        self.client_preprocess = client.get_preprocess()
        self.class_names = self.client_postprocess.load_class_names()
        self.count  = 0
        self.id_list_preds = []
        self.id_list_gts = []
        self.all_predictions = []
        self.all_groundtruths = []
        self.bag_processed = False
        self.gt_processed = False
        self.img_processed = False

        rospy.loginfo('Starting Prometheus Server on Port 7658')
        # a server which sends our metrics to the port 7658
        self.prometheus_server = start_http_server(7658)

        # sending metrics via prometheus client
        self.p_summary = Summary('precision', 'Precision of the Model')
        self.r_summary = Summary('recall', 'Recall of the Model')
        self.ap_summary = Summary('ap', 'Average Precision of the Model')
        self.f1_summary = Summary('fone', 'F1 Metric of the Model')
        self.ap_class_summary = Summary('ap_class', 'Average Precision per Class of the Model')

    def _register_inference(self):
        """
        register inference
        """
        # for GRPC channel
        if type(self.channel) == grpc_channel.GRPCChannel:
            self._set_grpc_channel_members()
        else:
            pass

        self.detection = rospy.Publisher(self.channel.params['pub_topic'], Image, queue_size=10)

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
        self.inference_topic = rospy.Subscriber(self.channel.params['sub_topic'], Image, self.image_callback)
        self.gt_topic = rospy.Subscriber(self.channel.params['gt_topic'], Detection2DArray, self.gt_callback)
        rospy.loginfo('Waiting until all the Rosbag messages are processed . . .')
        time.sleep(20)  # TODO wait until the inference is done, NO HARDCODING!
        statistics = self.calculate_metrics()

        rospy.loginfo('Sending Metrics to Prometheus')

        rospy.loginfo('Sent Metrics to Prometheus')

        rospy.spin()
        # if self.show_gt:
        #     self.visualize_gt()
        # if self.show_preds:
        #     self.visualize_pred()
        # self.calculate_metrics()

    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves
        # Arguments
            recall:    The recall curve (list)
            precision: The precision curve (list)
        # Returns
            Average precision, precision curve, recall curve
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    def ap_per_class(self, tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:  True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls:  Predicted object classes (nparray).
            target_cls:  True object classes (nparray).
            plot:  Plot precision-recall curve at mAP@0.5
            save_dir:  Plot save directory
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = (target_cls == c).sum()  # number of labels
            n_p = i.sum()  # number of predictions

            if n_p == 0 or n_l == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_l + 1e-16)  # recall curve
                r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

                # AP from recall-precision curve
                for j in range(tp.shape[1]):
                    ap[ci, j], mpre, mrec = self.compute_ap(recall[:, j], precision[:, j])
                    if plot and j == 0:
                        py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

        # Compute F1 (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + 1e-16)
        # if plot:
        #     plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        #     plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        #     plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        #     plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

        i = f1.mean(0).argmax()  # max F1 index
        return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

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

    def _scale_box_array(self, box, normalized=False):
        '''
        box: Bounding box generated for the image size (e.g. 512 x 512) expected by the model at triton server
        return: Scaled bounding box according to the input image from the ros topic.
        '''
        if normalized:
            # TODO make it dynamic with mc.Modelshape according to CHW or HWC
            xtl, xbr = box[0] * self.orig_size[1], box[2] * self.orig_size[1]
            ytl, ybr = box[1] * self.orig_size[0], box[3] * self.orig_size[0]
        else:
            xtl, xbr = box[:, 0] * (self.orig_size[1] / self.input_size[0]), \
                       box[:, 2] * (self.orig_size[1] / self.input_size[0])
            ytl, ybr = box[:, 1] * self.orig_size[0] / self.input_size[1], \
                       box[:, 3] * self.orig_size[0] / self.input_size[1]
        xtl = np.reshape(xtl, (len(xtl), 1))
        xbr = np.reshape(xbr, (len(xbr), 1))

        ytl = np.reshape(ytl, (len(ytl), 1))
        ybr = np.reshape(ybr, (len(ybr), 1))
        return np.concatenate((xtl, ytl, xbr, ybr, box[:, 4:6]), axis=1)

    # def _callback(self, msg):
    #     # rospy.loginfo('Image received...')
    #     cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    #     self.orig_size = cv_image.shape[0:2]
    #     self.orig_image = cv_image.copy()
    #     cv_image = cv2.resize(cv_image, (self.channel.input.shape[1], self.channel.input.shape[2]))
    #     self.image = self.client_preprocess.image_adjust(cv_image)
    #     if self.image is not None:
    #         self.channel.request.ClearField("inputs")
    #         self.channel.request.ClearField("raw_input_contents")  # Flush the previous image contents
    #         self.channel.request.inputs.extend([self.channel.input])
    #         self.channel.request.raw_input_contents.extend([self.image.tobytes()])
    #         self.channel.response = self.channel.do_inference() # perform the channel Inference
    #         self.prediction = self.client_postprocess.extract_boxes(self.channel.response)
    #
    #         for object in self.prediction[0]:  # predictions array has the order [x1,y1, x2,y2, confidence,
    #                                                                             # confidence, class ID]
    #             box = np.array(object[0:4], dtype=np.float32)
    #             box = self._scale_boxes(box, normalized=False)
    #             # if int(object[5]) == 0:
    #             #     color = (0, 255, 0)
    #             # else:
    #             #     color = (255, 0, 0)
    #             cv2.rectangle(self.orig_image,
    #                           pt1=(int(box[0]), int(box[1])),
    #                           pt2=(int(box[2]), int(box[3])),
    #                           color=(255, 0, 0),
    #                           thickness=3)
    #             cv2.putText(self.orig_image,
    #                         '{:.2f} {}'.format(object[-2], self.class_names[int(object[-1])]),
    #                         org=(int(box[0]), int(box[1])),
    #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                         fontScale=0.5,
    #                         thickness=2,
    #                         color=(0, 255, 0))
    #         # cv2.imshow('Prediction', cv2.cvtColor(self.orig_image, cv2.COLOR_RGB2BGR))
    #         # cv2.imshow('Prediction', cv2.cvtColor(self.orig_image, cv2.COLOR_RGB2BGR))
    #         # cv2.waitKey()
    #         self.msg_frame = self.br.cv2_to_imgmsg(self.orig_image, encoding="rgb8")
    #         self.msg_frame.header.stamp = rospy.Time.now()
    #         self.detection.publish(self.msg_frame)

    def image_callback(self, msg):
        self.count += 1
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.orig_size = cv_image.shape[0:2]
        self.orig_image = cv_image.copy()
        cv_image = cv2.resize(cv_image, (self.channel.input.shape[1], self.channel.input.shape[2]))
        self.image = self.client_preprocess.image_adjust(cv_image).astype(np.float32)
        if self.image is not None:
            self.channel.request.ClearField("inputs")
            self.channel.request.ClearField("raw_input_contents")   # Flush the previous image contents
            self.channel.request.inputs.extend([self.channel.input])
            self.channel.request.raw_input_contents.extend([self.image.tobytes()])
            self.channel.response = self.channel.do_inference()  # Inference
            self.prediction = self.client_postprocess.extract_boxes(self.channel.response)
            self.prediction = self._scale_box_array(self.prediction[0], normalized=False)

            # DEBUG
            # print('hold')
            # for object in self.prediction[0]:
            #     box = np.array(object[0:4], dtype=np.float32)
            #     box = self._scale_boxes(box, normalized=False)
            #     cv2.rectangle(self.orig_image,
            #                   pt1=(int(box[0]), int(box[1])),
            #                   pt2=(int(box[2]), int(box[3])),
            #                   color=(255, 0, 0),
            #                   thickness=2)
            #     # cv2.putText(self.orig_image,
            #     #             '{:.2f} {}'.format(self.scores[idx], self.class_names[int(self.class_ids[idx])]),
            #     #             org=(int(box[0]), int(box[1])),
            #     #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #     #             fontScale=0.5,
            #     #             thickness=2,
            #     #             color=(0, 255, 0))
            #     # cv2.imshow('prediction', self.orig_image)
            #     # cv2.waitKey()
            # cv2.imshow('prediction', self.orig_image)
            # cv2.waitKey()
            if msg.header.seq not in self.id_list_preds:
                self.id_list_preds.append(msg.header.seq)
                self.all_predictions.append({
                    'header': msg.header.seq,
                    'prediction': self.prediction,
                })
            elif msg.header.seq in self.id_list_preds:
                # this ros bag is done now stop spinning it.
                rospy.loginfo('Closing the subscriber topic: {}'.format(self.channel.params['sub_topic']))
                self.img_processed = True
                self.inference_topic.unregister()
                # rospy.signal_shutdown('Closing rospy for postprocessing!')

    def gt_callback(self, msg):
        # rospy.loginfo('Detection received...')
        # rospy.loginfo('Message count {}'.format(self.count))
        # self.count +=1
        gt = {
            'header': msg.detections[0].source_img.header.seq,
            'ground_truths': None
        }

        if msg.detections[0].source_img.header.seq not in self.id_list_gts:
            self.id_list_gts.append(msg.detections[0].source_img.header.seq)
            gts = np.zeros((len(msg.detections), 6), dtype=float)

            for detection, idx in zip(msg.detections, range(len(msg.detections))):
                gts[idx, :] = [detection.bbox.center.x,
                               detection.bbox.center.y,
                               detection.bbox.size_x,
                               detection.bbox.size_y,
                               detection.results[0].score,
                               detection.results[0].id]
            # convert to x1,y1 and x2,y2 format
            gts = self.client_postprocess.xywh2xyxy(gts)
            # print('hold')
            # DEBUG
            # for object in gts:
            #     box = np.array(object[0:4], dtype=np.float32)
            #     # box = self._scale_boxes(box, normalized=False)
            #     cv2.rectangle(self.orig_image,
            #                   pt1=(int(box[0]), int(box[1])),
            #                   pt2=(int(box[2]), int(box[3])),
            #                   color=(255, 0, 0),
            #                   thickness=2)
            #     # cv2.putText(self.orig_image,
            #     #             '{:.2f} {}'.format(self.scores[idx], self.class_names[int(self.class_ids[idx])]),
            #     #             org=(int(box[0]), int(box[1])),
            #     #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #     #             fontScale=0.5,
            #     #             thickness=2,
            #     #             color=(0, 255, 0))
            #     # cv2.imshow('prediction', self.orig_image)
            #     # cv2.waitKey()
            # cv2.imshow('prediction', self.orig_image)
            # cv2.waitKey()
            gt['ground_truths'] = gts
            self.all_groundtruths.append(gt)
        elif msg.detections[0].source_img.header.seq in self.id_list_gts:
            rospy.loginfo('Closing the subscriber topic: {}'.format(self.channel.params['gt_topic']))
            self.gt_processed = True
            self.gt_topic.unregister()

    def calculate_metrics(self):
        import torch
        names = {0: 'Weeds', 1: 'Maize'}
        aggregated_stats = []

        for msg_number in range(len(self.all_groundtruths)):
            rospy.loginfo('Processing message number {} out of {} . . .'.format(msg_number+1, len(self.all_groundtruths)))
            if self.img_processed and self.gt_processed:    # Processing DONE!!!!
                iou = self.client_postprocess.box_iou(
                    torch.tensor(self.all_groundtruths[msg_number]['ground_truths'][:, :4]),
                    torch.tensor(self.all_predictions[msg_number]['prediction'][:, :4]))
                iouv = torch.tensor([0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95])
                labelsn = torch.unsqueeze(torch.tensor(self.all_groundtruths[msg_number]['ground_truths'][:, 5]), 1)
                detections = torch.tensor(self.all_predictions[msg_number]['prediction'][:, 5])
                correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
                x = torch.where((iou >= iouv[0]) & (labelsn == detections))  # IoU above threshold and classes match
                if x[0].shape[0]:
                    matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                        1).cpu().numpy()  # [label, detection, iou]
                    if x[0].shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    matches = torch.Tensor(matches)
                    correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv

                    stats = [(correct.cpu(),
                                  self.all_predictions[msg_number]['prediction'][:, 4],
                                  self.all_predictions[msg_number]['prediction'][:, 5],
                                  self.all_groundtruths[msg_number]['ground_truths'][:, 5])]

                    # TODO publish summary of aggregated_stats to graphana
                    aggregated_stats.append(stats)

                    stats = [np.concatenate(x, 0) for x in zip(*stats)]
                    p, r, ap, f1, ap_class = self.ap_per_class(*stats, names=names)

                    # sending information to metric objects
                    # the metrics are a list, and each value has to be sent separately
                    [self.p_summary.observe(_) for _ in p]
                    [self.r_summary.observe(_) for _ in r]
                    #[self.ap_summary.observe(_) for _ in ap]
                    [self.f1_summary.observe(_) for _ in f1]
                    [self.ap_class_summary.observe(_) for _ in ap_class]

        return aggregated_stats
