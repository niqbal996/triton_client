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
import numpy as np
import sys
import rospy
import yaml

import grpc
import cv2

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

from utils.preprocess import parse_model, model_dtype_to_np, requestGenerator, image_adjust
from utils.postprocess import extract_boxes_triton, load_class_names
from utils.evaluate_predictions import EvaluatorNode

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-a',
                        '--async',
                        dest="async_set",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use asynchronous inference API')
    parser.add_argument('--streaming',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use streaming inference API')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=False,
                        default="YOLOv5n",
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
    parser.add_argument('-c',
                        '--classes',
                        type=int,
                        required=False,
                        default=80,
                        help='Number of class results to report. Default is 1.')
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
        '--image-src',
        type=str,
        choices=['ros', 'local'],
        required=False,
        default='ros',
        help='Source of input images to run inference on. Default is ROS image topic')
    FLAGS = parser.parse_args()
    rospy.init_node('ros_infer')
    param_file = rospy.get_param('client_parameter_file', './data/client_parameter.yaml')
    with open(param_file) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    # Create gRPC stub for communicating with the server
    # TODO! Make dynamic message length depending upon the size of input image.
    channel = grpc.insecure_channel(param['grpc_channel'], options=[
                                   ('grpc.max_send_message_length', FLAGS.batch_size*8568044),
                                   ('grpc.max_receive_message_length', FLAGS.batch_size*8568044),
                                    ])
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    class_names = load_class_names()

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    metadata_request = service_pb2.ModelMetadataRequest(
        name=FLAGS.model_name, version=FLAGS.model_version)
    metadata_response = grpc_stub.ModelMetadata(metadata_request)

    config_request = service_pb2.ModelConfigRequest(name=FLAGS.model_name,
                                                    version=FLAGS.model_version)
    config_response = grpc_stub.ModelConfig(config_request)

    input_name, output_name, c, h, w, format, dtype = parse_model(
        metadata_response, config_response.config)

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []

    # Send request
    if FLAGS.streaming and not FLAGS.ros_topic:
        for response in grpc_stub.ModelStreamInfer(
                requestGenerator(input_name, output_name, c, h, w, format,
                                 dtype, FLAGS, result_filenames)):
            responses.append(response)
    elif FLAGS.image_src == 'ros':
        ros_node = EvaluatorNode(grpc_stub,
                                 input_name,
                                 output_name,
                                 param,
                                 FLAGS,
                                 dtype,
                                 c, h, w,
                                 visualize_gt=True,
                                 visualize_pred=True)
        ros_node.start_evaluation()
    else:
        for request in requestGenerator(input_name, output_name, c, h, w,
                                        format, dtype, FLAGS, result_filenames):
            if not FLAGS.async_set:
                # requests.append(request)
                response = grpc_stub.ModelInfer(request)
                prediction = extract_boxes_triton(response)
            else:
                requests.append(grpc_stub.ModelInfer.future(request))