import grpc
from abc import ABC, abstractmethod
from tritonclient.grpc import service_pb2, service_pb2_grpc


class BaseCommunicator(ABC):

    def __init__(self, params, FLAGS,client):
        self._mode = None
        self.params = params
        self.FLAGS = FLAGS
        self._stub = None
        self._meta_data = {}
        self._register_mode()
        self.client = client
        self.client_preprocess = client.get_preprocess()
        self.client_postprocess = client.get_postprocess()

    @abstractmethod
    def _register_communicator(self):
        """ Implement different communicator"""

    def _register_mode(self):
        """
        register mode of communication, grpc or http
        """
        # TODO! Make dynamic message length depending upon the size of input image.
        channel = grpc.insecure_channel(self.params['grpc_channel'], options=[
            ('grpc.max_send_message_length', self.FLAGS.batch_size * 8568044),
            ('grpc.max_receive_message_length', self.FLAGS.batch_size * 8568044),
        ])
        grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
        self._stub = grpc_stub

        self._register_metadata()

    def _register_metadata(self):
        """
        register metadata for the particular mode
        """
        self._meta_data["metadata_request"] = service_pb2.ModelMetadataRequest(
            name=self.FLAGS.model_name, version=self.FLAGS.model_version)
        self._meta_data["metadata_response"] = self._stub.ModelMetadata(self._meta_data["metadata_request"])

        self._meta_data["config_request"] = service_pb2.ModelConfigRequest(name=self.FLAGS.model_name,
                                                             version=self.FLAGS.model_version)
        self._meta_data["config_response"]= self._stub.ModelConfig(self._meta_data["config_request"])

    def get_mode(self):
        """
        get mode of communicator
        """
        return self._stub

    def get_metadata(self):
        """
        get metadata of model
        """
        return self._meta_data
