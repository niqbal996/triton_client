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
from PIL import Image
import os
import sys
import struct

import grpc
import cv2

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

from utils.preprocess import parse_model, model_dtype_to_np, requestGenerator, image_adjust
from utils.postprocess import extract_boxes_triton, load_class_names

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
                        required=True,
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
                        default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument(
        '-s',
        '--scaling',
        type=str,
        choices=['NONE', 'INCEPTION', 'VGG', 'COCO'],
        required=False,
        default='NONE',
        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('image_filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()

    # Create gRPC stub for communicating with the server
    # NOTE! Depending upon the image dimensions, the message length has to be adjusted. This works for 512 x 512 x 3
    channel = grpc.insecure_channel(FLAGS.url, options=[
                                   ('grpc.max_send_message_length', 5419071),
                                   ('grpc.max_receive_message_length', 5419071),
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
    if FLAGS.streaming:
        for response in grpc_stub.ModelStreamInfer(
                requestGenerator(input_name, output_name, c, h, w, format,
                                 dtype, FLAGS, result_filenames)):
            responses.append(response)
    else:
        for request in requestGenerator(input_name, output_name, c, h, w,
                                        format, dtype, FLAGS, result_filenames):
            if not FLAGS.async_set:
                requests.append(request)
                responses.append(grpc_stub.ModelInfer(request))
            else:
                requests.append(grpc_stub.ModelInfer.future(request))

    # For async, retrieve results according to the send order
    if FLAGS.async_set:
        for request in requests:
            responses.append(request.result())

    error_found = False
    idx = 0
    predictions = []
    for response in responses:
        if FLAGS.streaming:
            if response.error_message != "":
                error_found = True
                print(response.error_message)
            else:
                predictions.append(extract_boxes_triton(response.infer_response, result_filenames[idx],
                            FLAGS.batch_size))
        else:
            predictions.append(extract_boxes_triton(response, result_filenames[idx], FLAGS.batch_size))
        idx += 1
    # TODO add publish ROS message flag to display detections instead of always true.
    # TODO the file ordering is not consistent with the inference results. TRITON must maintain the order. NOT CHAOS!!!
    if True:
        for file, pred in zip(result_filenames, predictions):
            cv_image = cv2.imread(file[0])
            # cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv_image = cv2.resize(cv_image, (w, h), interpolation=cv2.INTER_LINEAR)
            for object in pred[0]:  # predictions array has the order [x1,y1, x2,y2, confidence, confidence, class ID]
                box = np.array(object[0:4], dtype=np.float32) * w
                cv2.rectangle(cv_image,
                              pt1=(int(box[0]), int(box[1])),
                              pt2=(int(box[2]), int(box[3])),
                              color=(0, 255, 0),
                              thickness=1)
                cv2.putText(cv_image,
                            '{:.2f} {}'.format(object[-2], class_names[int(object[-1])]),
                            org=(int(box[0]), int(box[1])),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=2,
                            color=(0, 255, 0))

            cv2.imshow('prediction', cv_image)
            cv2.waitKey()
    if error_found:
        sys.exit(1)
