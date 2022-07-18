import tritonclient.grpc.model_config_pb2 as mc

from .base_client import Client
from .preprocess import PointpillarPreprocess
from .postprocess import PointPillarPostprocess

class Pointpillars_client(Client):
    """

    """
    def __init__(self, ):
        super().__init__()

    def register_client(self, clienttype, client):
        """
        Implement the method to register the client for
        """
        self._clients[clienttype] = client

    def get_preprocess(self):
        return PointpillarPreprocess()

    def get_postprocess(self):
        return PointPillarPostprocess()

    def parse_model(self, model_metadata, model_config):
        if len(model_metadata.inputs) != 3:     # voxels, coords, numpoints
            raise Exception("expecting 3 input, got {}".format(
                len(model_metadata.inputs)))
        if len(model_metadata.outputs) != 3:    # bbox_preds, dir_scores, scores
            raise Exception("expecting 3 output, got {}".format(
                len(model_metadata.outputs)))

        input_metadata = [input for input in model_metadata.inputs]
        input_config = model_config.input[0]
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
        # if len(input_metadata.shape) != expected_input_dims:
        #     raise Exception(
        #         "expecting input to have {} dimensions, model '{}' input has {}".
        #             format(expected_input_dims, model_metadata.name,
        #                    len(input_metadata.shape)))
        # TODO with or without Reflectance
        # if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
        #         (input_config.format != mc.ModelInput.FORMAT_NHWC)):
        #     raise Exception("unexpected input format " +
        #                     mc.ModelInput.Format.Name(input_config.format) +
        #                     ", expecting " +
        #                     mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
        #                     " or " +
        #                     mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

        # if input_config.format == mc.ModelInput.FORMAT_NHWC:
        #     h = input_metadata.shape[1 if input_batch_dim else 0]
        #     w = input_metadata.shape[2 if input_batch_dim else 1]
        #     c = input_metadata.shape[3 if input_batch_dim else 2]
        # else:
        #     c = input_metadata.shape[1 if input_batch_dim else 0]
        #     h = input_metadata.shape[2 if input_batch_dim else 1]
        #     w = input_metadata.shape[3 if input_batch_dim else 2]

        return ([input.name for input in input_metadata], 
                [output.name for output in output_metadata],
                [input.datatype for input in input_metadata])