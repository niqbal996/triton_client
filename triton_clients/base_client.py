from __future__ import annotations
from abc import ABC, abstractmethod


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

    @abstractmethod
    def get_client(self,clienttype):
        """
        """

    @abstractmethod
    def get_postprocess(self):
        """
        """
    @abstractmethod
    def get_preprocess(self):
        """"""

    @abstractmethod
    def parse_model(self):
        """
        """


