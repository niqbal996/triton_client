import grpc
from abc import ABC, abstractmethod
from tritonclient.grpc import service_pb2, service_pb2_grpc
from communicator.channel import base_channel

class BaseInference(ABC):

    def __init__(self, channel:base_channel.BaseChannel, client):
        self.channel = channel
        self.client = client

    def load_class_names(self, namesfile='./data/crop.names'):
        class_names = []
        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names



