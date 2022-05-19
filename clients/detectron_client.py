from .base_client import Client
from ..preprocess import FCOSpreprocess
from ..postprocess import FCOSpostprocess
class FCOS_client(Client):
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
        return FCOSpreprocess()

    def get_postprocess(self):
        return FCOSpostprocess()