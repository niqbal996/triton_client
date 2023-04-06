#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
# import numpy as np
# import sys
import rospy
import yaml

# import grpc
# import cv2
#
# from tritonclient.grpc import service_pb2, service_pb2_grpc
# import tritonclient.grpc.model_config_pb2 as mc
#
# from utils.preprocess import parse_model, model_dtype_to_np, requestGenerator, image_adjust
# from utils.postprocess import extract_boxes_triton, load_class_names
# from utils.ros_input import RealSenseNode

from communicator import RosInference, EvaluateInference
from communicator.channel import grpc_channel, seerep_channel
from clients import Yolov5client, FCOS_client

clients = {
    'YOLOv5nCROP': Yolov5client,
    'YOLOv5nCOCO': Yolov5client,
    'FCOS_detectron':FCOS_client,
    # 'second_iou':Pointpillars_client,
    # more clients can be added
}


FLAGS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=False,
                        default="YOLOv5nCROP",
                        help='Name of model')
    parser.add_argument(
        '-x',
        '--model-version',
        type=str,
        required=False,
        default="",
        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument(
        '-s',
        '--scaling',
        type=str,
        choices=['NONE', 'INCEPTION', 'VGG', 'COCO'],
        required=False,
        default='COCO',
        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument(
        '-i',
        '--imagesrc',
        type=str,
        choices=['ros', 'local', 'seerepfb'],
        required=False,
        default='ros',
        help='Source of input images to run inference on. Default is ROS image topic')
    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = parse_args()
    rospy.init_node('ros_infer')
    param_file = rospy.get_param('client_parameter_file', './data/client_parameter.yaml')
    with open(param_file) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    client = clients[FLAGS.model_name](model_name=FLAGS.model_name)
    # client = FCOS_client(model_name='FCOS_detectron')

    #define channel
    channel = grpc_channel.GRPCChannel(param, FLAGS)

    #define inference
    #inference = RosInference(channel, client)
    #inference.start_inference()
    evaluation = EvaluateInference(channel, client)
    evaluation.start_inference(FLAGS.imagesrc, FLAGS.model_name)
