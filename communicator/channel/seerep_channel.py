from communicator.channel.base_channel import BaseChannel

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc


class SEEREPChannel(BaseChannel):
    """
    A SEEREPChannel is establishes a connection between the triton client and SEEREP.
    """

    def __init__(self,params,FLAGS):
        super().__init__(params,FLAGS)
        self._meta_data = {}
        self._grpc_stub = None
        self._grpc_stubmeta = None
        self._builder = None
        self._projectid = None

        self.register_channel() # register and initialise the stub
        self._grpc_metadata() #

    def register_channel(self):
        """
         register grpc triton channel
        """
        # server with certs
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(__location__, 'tls.pem'), 'rb') as f:
            root_cert = f.read()
        server = "seerep.robot.10.249.3.13.nip.io:32141"
        creds = grpc.ssl_channel_credentials(root_cert)
        channel = grpc.secure_channel(server, creds)

        self._grpc_stub  = imageService.ImageServiceStub(channel)
        self._grpc_stubmeta = stubMeta = metaOperations.MetaOperationsStub(channel)  
        self._builder = self.init_query()
        self._projectid = self.retrieveproject()

    def fetch_channel(self):
        """
        return grpc stub
        """
        return self._grpc_stub

    def _grpc_metadata(self):
        """
        TODO Figure out if this is needed for SEEREP
        Initiate all meta data required for models
        """
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        self._meta_data["metadata_request"] = service_pb2.ModelMetadataRequest(
            name=self.FLAGS.model_name, version=self.FLAGS.model_version)
        self._meta_data["metadata_response"] = self._grpc_stub.ModelMetadata(self._meta_data["metadata_request"])

        self._meta_data["config_request"] = service_pb2.ModelConfigRequest(name=self.FLAGS.model_name,
                                                                           version=self.FLAGS.model_version)
        self._meta_data["config_response"] = self._grpc_stub.ModelConfig(self._meta_data["config_request"])

        # set
        self._set_grpc_members()

    def get_metadata(self):
        """
        return meta_data dictionary form
        @rtype: dictionary
        """
        return self._meta_data

    def _set_grpc_members(self):
        """
        set essential grpc members
        """
        self.input = service_pb2.ModelInferRequest().InferInputTensor()
        self.request = service_pb2.ModelInferRequest()
        self.request.model_name = self.FLAGS.model_name
        self.request.model_version = self.FLAGS.model_version
        self.output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()

    def do_inference(self):
        """
        inference based on grpc_stud
        @return: inference of grpc
        """
        return self._grpc_stub.ModelInfer(self.request)

    def retrieve_project(self, projname):
        '''
            WHAT THE FUCK
        '''
    for i in range(response.ProjectsLength()):
        print(response.Projects(i).Name().decode("utf-8") + " " + response.Projects(i).Uuid().decode("utf-8"))
        if response.Projects(i).Name().decode("utf-8") == projname:
            projectuuid = response.Projects(i).Uuid().decode("utf-8")

        return projectuuid


    def init_query(self):
        builder = flatbuffers.Builder(1024)
        Empty.Start(builder)
        emptyMsg = Empty.End(builder)
        builder.Finish(emptyMsg)
        buf = builder.Output()

        Query.Start(builder)

        return builder

    def add_boundingbox(start_coord, end_coord):
        '''
        Add a bounding box to the query builder
        Args:
            start_coord : An interable of the X, Y and Z start co ordinates of the point, in this order.
            end_coord : An interable of the X, Y and Z end co ordinates of the point, in this order.
        '''
        
        Point.Start(self._builder)
        Point.AddX(self._builder, start_coord[0])
        Point.AddY(self._builder, start_coord[1])
        Point.AddZ(self._builder, start_coord[2])
        pointMin = Point.End(self._builder)

        Point.Start(self._builder)
        Point.AddX(self._builder, end_coord[0])
        Point.AddY(self._builder, end_coord[1])
        Point.AddZ(self._builder, end_coord[2])
        pointMax = Point.End(self._builder)

        frameId = builder.CreateString("map")
        Header.Start(builder)
        Header.AddFrameId(builder, frameId)
        header = Header.End(builder)

        Boundingbox.Start(builder)
        Boundingbox.AddPointMin(builder, pointMin)
        Boundingbox.AddPointMax(builder, pointMax)
        Boundingbox.AddHeader(builder, header)
        boundingbox = Boundingbox.End(builder)

        Query.AddBoundingbox(builder, boundingbox)

    def add_timestamp(self, starttime, endtime):
        '''
        Add a time range to the query builder
        Args:
            starttime : Start time as an int
            endtime : End time as an int
        '''

        Timestamp.Start(self._builder)
        Timestamp.AddSeconds(self._builder, starttime)
        Timestamp.AddNanos(self._builder, 0)
        timeMin = Timestamp.End(self._builder)

        Timestamp.Start(self._builder)
        Timestamp.AddSeconds(self._builder, endtime)
        Timestamp.AddNanos(self._builder, 0)
        timeMax = Timestamp.End(self._builder)

        TimeInterval.Start(self._builder)
        TimeInterval.AddTimeMin(self._builder, timeMin)
        TimeInterval.AddTimeMax(self._builder, timeMax)
        timeInterval = TimeInterval.End(self._builder)

        projectuuidString = self._builder.CreateString(self._projectuuid)
        Query.StartProjectuuidVector(self._builder, 1)
        builder.PrependUOffsetTRelative(projectuuidString)
        projectuuidMsg = self._builder.EndVector()

        Query.AddTimeinterval(builder, timeInterval)

    def add_label(self, label):
        label = builder.CreateString("1")
        Query.StartLabelVector(builder, 1)
        builder.PrependUOffsetTRelative(label)
        labelMsg = builder.EndVector()

        Query.AddLabel(builder, labelMsg)

    def run_query(self):
        queryMsg = Query.End(builder)

        builder.Finish(queryMsg)
        buf = builder.Output()

        for responseBuf in stub.GetImage(bytes(buf)):
            response = Image.Image.GetRootAs(responseBuf)
            print("uuidmsg: " + response.Header().UuidMsgs().decode("utf-8"))
            print("first label: " + response.LabelsBb(0).LabelWithInstance().Label().decode("utf-8"))
            print(
                "first BoundingBox (Xmin,Ymin,Xmax,Ymax): "
                + str(response.LabelsBb(0).BoundingBox().PointMin().X())
                + " "
                + str(response.LabelsBb(0).BoundingBox().PointMin().Y())
                + " "
                + str(response.LabelsBb(0).BoundingBox().PointMax().X())
                + " "
                + str(response.LabelsBb(0).BoundingBox().PointMax().Y())
            )













