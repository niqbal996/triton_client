# from __future__ import annotations
from abc import ABC, abstractmethod
import tritonclient.grpc.model_config_pb2 as mc

class Client(ABC):
    """
        Declares the functionality for several clients for triton server
    """

    def __init__(self):
        self._clients = {}

    @abstractmethod
    def register_client(self,clienttype,client):
        """
        Implement the method to register the client for
        """

    # # @abstractmethod
    # def get_client(self,clienttype):
    #     """
    #     """

    @abstractmethod
    def get_postprocess(self):
        """
        """
    @abstractmethod
    def get_preprocess(self):
        """"""

    def parse_model(self, model_metadata, model_config):
        """
            Parse the model based on the model configuration file on the triton server
            This works for most image based models. For 3D detectors, the method is overridden.
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


