from vision_msgs.msg import Detection2DArray
import numpy as np


import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose, PoseWithCovariance

# onnx inference imports
import torch
# Further imports
import cv2  # NOTE! Importing cv_bridge before cv2 on adves box gives error.
import yaml
import numpy as np
import os

def yolo2vision_msg(yolo_prediction, output):
    pass

# detection_array = Detection2DArray()
# detection = Detection2D()
# object_hypothesis = ObjectHypothesisWithPose()
# dummy_pose = PoseWithCovariance()
# for i, det in enumerate(pred):
#     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig.shape).round()
#     for *xyxy, conf, cls in reversed(det):
#         theta = 0.0  # 2D boxes are always horizontally aligned for YOLO
#         box = [int(np.array(i.cpu())) for i in xyxy]
#         w, h = int(box[2] - box[0]), int(box[3] - box[1])
#         # Create a Detection2D message
#         detection.header = msg.header
#         # 2D pose
#         detection.bbox.center.x = int(box[0] + w / 2)
#         detection.bbox.center.y = int(box[1] + h / 2)
#         detection.bbox.center.theta = theta
#         # size of the box
#         detection.bbox.size_x = w
#         detection.bbox.size_y = h
#         # Create an object hypothesis message WITH DUMMY pose
#         object_hypothesis.id = int(cls)
#         object_hypothesis.score = round(conf.item(), 4)
#         # NOTE!!! There is no pose prediction for the 2D boxes and all are horizontally aligned hence dummy values.
#         object_hypothesis.pose = dummy_pose
#
#         detection.results.append(object_hypothesis)
#         # Single Detection2D message is ready
#         detection_array.detections.append(detection)
#         # reset
#         detection = Detection2D()
# detection_array.header = msg.header
# detection_topic.publish(detection_array)