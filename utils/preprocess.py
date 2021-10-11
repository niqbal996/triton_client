import sys

import numpy as np
import os
import cv2
import rospy
from PIL import Image
import yaml

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

dir_path = os.path.dirname(os.path.realpath(__file__))

def model_dtype_to_np(model_dtype):
    if model_dtype == "BOOL":
        return bool
    elif model_dtype == "INT8":
        return np.int8
    elif model_dtype == "INT16":
        return np.int16
    elif model_dtype == "INT32":
        return np.int32
    elif model_dtype == "INT64":
        return np.int64
    elif model_dtype == "UINT8":
        return np.uint8
    elif model_dtype == "UINT16":
        return np.uint16
    elif model_dtype == "FP16":
        return np.float16
    elif model_dtype == "FP32":
        return np.float32
    elif model_dtype == "FP64":
        return np.float64
    elif model_dtype == "BYTES":
        return np.dtype(object)
    return None

def load_class_names(namesfile='{}/data/crop.names'.format(dir_path)):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """

    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    # if len(model_metadata.outputs) != 1:
    #     raise Exception("expecting 1 output, got {}".format(
    #         len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    # TODO multiple outputs: such as boxes, configs, classes
    output_metadata = [output for output in model_metadata.outputs]
    # input_config.format = 2
    # if output_metadata.datatype != "FP32":
    #     raise Exception("expecting output datatype to be FP32, model '" +
    #                     model_metadata.name + "' output type is " +
    #                     output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    # for dim in output_metadata.shape:
    #     if output_batch_dim:
    #         output_batch_dim = False
    #     elif dim > 1:
    #         non_one_cnt += 1
    #         if non_one_cnt > 1:
    #             raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = (model_config.max_batch_size > 0)
    input_batch_dim = False
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
                   len(input_metadata.shape)))

    if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
        (input_config.format != mc.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " +
                        mc.ModelInput.Format.Name(input_config.format) +
                        ", expecting " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                        " or " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (input_metadata.name, [output.name for output in output_metadata], c, h, w,
            input_config.format, input_metadata.datatype)


def image_adjust(img, format, dtype, c, h, w, scaling, pil=False):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    #np.set_printoptions(threshold='nan')
    if pil:
        if c == 1:
            sample_img = img.convert('L')
        else:
            sample_img = img.convert('RGB')

        resized_img = sample_img.resize((w, h), Image.BILINEAR)
        resized = np.array(resized_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        npdtype = model_dtype_to_np(dtype)
        typed = resized.astype(npdtype)

        if scaling == 'INCEPTION':
            scaled = (typed / 127.5) - 1
        elif scaling == 'VGG':
            if c == 1:
                scaled = typed - np.asarray((128,), dtype=npdtype)
            else:
                scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
        elif scaling == 'COCO':
            scaled = typed / 255.0
        else:
            scaled = typed

        # Swap to CHW if necessary
        if format == mc.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled
        # Channels are in RGB order. Currently model configuration data
        # doesn't provide any information as to other channel orderings
        # (like BGR) so we just assume RGB.
        return ordered
    else:
        img_in = cv2.imread(img)
        img_in = cv2.resize(img_in, (w, h), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        return img_in

def image_adjust_ros(cv_image):
    pad = np.zeros((16, 1280, 3), dtype=np.uint8)
    cv_image = np.concatenate((cv_image, pad), axis=0)
    # orig = cv_image.copy()
    cv_image = np.transpose(cv_image, (2, 0, 1)).astype(np.float32)
    cv_image = np.expand_dims(cv_image, axis=0)
    cv_image /= 255.0

def requestGenerator(input_name, output_name, c, h, w, format, dtype, FLAGS,
                     result_filenames):
    request = service_pb2.ModelInferRequest()
    request.model_name = FLAGS.model_name
    request.model_version = FLAGS.model_version

    if FLAGS.image_filename:
        filenames = []
        if os.path.isdir(FLAGS.image_filename):
            filenames = [
                os.path.join(FLAGS.image_filename, f)
                for f in os.listdir(FLAGS.image_filename)
                if os.path.isfile(os.path.join(FLAGS.image_filename, f)) and
                   (os.path.join(FLAGS.image_filename, f).endswith('jpg') or
                    os.path.join(FLAGS.image_filename, f).endswith('png'))
            ]
        else:
            filenames = [
                FLAGS.image_filename,
            ]

        filenames.sort()
        image_data = []
        filenames = filenames[-10:]
        for filename in filenames:
            # img = Image.open(filename)
            image_data.append(image_adjust(filename, format, dtype, c, h, w,
                                           FLAGS.scaling))

    else:
        print("No input data specified")
        sys.exit()
    output0 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output0.name = output_name[0]
    output1 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output1.name = output_name[1]
    request.outputs.extend([output0, output1])
    # request.outputs.extend([output0])
    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = input_name
    input.datatype = dtype
    # if format == mc.ModelInput.FORMAT_NHWC:
    #     input.shape.extend([FLAGS.batch_size, h, w, c])
    # else:
    #     input.shape.extend([FLAGS.batch_size, c, h, w])
    if format == mc.ModelInput.FORMAT_NHWC:
        input.shape.extend([h, w, c])
    else:
        input.shape.extend([c, h, w])

    # Preprocess image into input data according to model requirements
    # Preprocess the images into input data according to model
    # requirements

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    image_idx = 0
    last_request = False
    while not last_request:
        input_bytes = None
        input_filenames = []
        request.ClearField("inputs")
        request.ClearField("raw_input_contents")
        for idx in range(FLAGS.batch_size):
            input_filenames.append(filenames[image_idx])
            if input_bytes is None:
                input_bytes = image_data[image_idx].tobytes()
            else:
                input_bytes += image_data[image_idx].tobytes()

                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True

        request.inputs.extend([input])
        result_filenames.append(input_filenames)
        request.raw_input_contents.extend([input_bytes])
        yield request

