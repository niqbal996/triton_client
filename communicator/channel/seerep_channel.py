
from communicator.channel.base_channel import BaseChannel
#from base_channel import BaseChannel

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

import os
import sys
import cv2
import numpy as np

import flatbuffers
import grpc
from seerep.fb import BoundingBoxes2DLabeledStamped, Boundingbox, Empty, Header, Image, Point, ProjectInfos, Query, TimeInterval, Timestamp

from seerep.fb import image_service_grpc_fb as imageService
from seerep.fb import meta_operations_grpc_fb as metaOperations

import communicator.channel.util_fb as util_fb

import uuid


class SEEREPChannel():
    """
    A SEEREPChannel is establishes a connection between the triton client and SEEREP.
    """
    def __init__(self, project_name='testproject', socket='agrigaia-ur.ni.dfki:9090'):
        self._meta_data = {}
        self._grpc_stub = None
        self._grpc_stubmeta = None
        self._builder = None
        self._projectid = None
        self._msguuid = None
        self.socket = socket
        self.projname = project_name

        # register and initialise the stub
        self.channel = self.make_channel(secure=False)
        self.register_channel()
        #self._grpc_metadata() #

    def make_channel (self, secure=False):
        # server with certs
        if secure:
            __location__ = os.path.realpath(
                os.path.join(os.getcwd(), os.path.dirname(__file__)))
            with open(os.path.join(__location__, 'tls.pem'), 'rb') as f:
                root_cert = f.read()
            creds = grpc.ssl_channel_credentials(root_cert)

            channel = grpc.secure_channel(self.socket, creds) # use with non-local deployment

        else:
            channel = grpc.insecure_channel(self.socket) # use with local deployment

        return channel

    def register_channel(self):
        """
         register grpc triton channel
         socket: String, Port and IP address of seerep server
         seerep.robot.10.249.3.13.nip.io:32141
        """
        self._grpc_stub  = imageService.ImageServiceStub(self.channel)
        self._grpc_stubmeta = metaOperations.MetaOperationsStub(self.channel)  
        self._builder = self.init_builder()
        self._msgUuid = None
        self._projectid = self.retrieve_project(self.projname, log=True)

    def secondary_channel(self):
        """
         Establish another channel for sending
         register grpc triton channel
         socket: String, Port and IP address of seerep server
         seerep.robot.10.249.3.13.nip.io:32141
        """
        grpc_stub  = imageService.ImageServiceStub(self.channel)
        grpc_stubmeta = metaOperations.MetaOperationsStub(self.channel)  
        builder = self.init_builder()
        projectid = self.retrieve_project(self.projname, log=False)

        return (grpc_stub, grpc_stubmeta, builder, projectid)

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

    def perform_inference(self):
        """
        inference based on grpc_stud
        @return: inference of grpc
        """
        return self._grpc_stub.ModelInfer(self.request)

    def retrieve_project(self, projname, log=False):
        '''
        '''
        Empty.Start(self._builder)
        emptyMsg = Empty.End(self._builder)
        self._builder.Finish(emptyMsg)
        buf = self._builder.Output()

        responseBuf = self._grpc_stubmeta.GetProjects(bytes(buf))
        response = ProjectInfos.ProjectInfos.GetRootAs(responseBuf)

        for i in range(response.ProjectsLength()):
            if log==True:
                try:
                    print(response.Projects(i).Name().decode("utf-8") + " " + response.Projects(i).Uuid().decode("utf-8"))
                    if response.Projects(i).Name().decode("utf-8") == projname:
                        projectuuid = response.Projects(i).Uuid().decode("utf-8")
                        print("[FOUND PROJ] ", projectuuid)
                except Exception as e:
                    print(e)
            else:
                try:
                    projectuuid = response.Projects(i).Uuid().decode("utf-8")
                except Exception as e:
                    print(e)
        return projectuuid

    def string_to_fbmsg (self, projectuuid):
        projectuuidString = self._builder.CreateString(projectuuid)
        '''
        Query.StartProjectuuidVector(self._builder, 1)
        self._builder.PrependUOffsetTRelative(projectuuidString)
        projectuuidMsg = self._builder.EndVector()

        return projectuuidMsg
        '''
        return projectuuidString

    def init_builder(self):
        builder = flatbuffers.Builder(1024)
        
        return builder

    def gen_boundingbox(self, start_coord, end_coord):
        '''
        Add a bounding box to the query builder
        Args:
            start_coord : An interable of the X, Y and Z start co ordinates of the point, in this order.
            end_coord : An interable of the X, Y and Z end co ordinates of the point, in this order.
        '''
        
        Point.Start(self._builder)
        Point.AddX(self._builder, start_coord[0])
        Point.AddY(self._builder, start_coord[1])
        #Point.AddZ(self._builder, start_coord[2])
        pointMin = Point.End(self._builder)

        Point.Start(self._builder)
        Point.AddX(self._builder, end_coord[0])
        Point.AddY(self._builder, end_coord[1])
        #Point.AddZ(self._builder, end_coord[2])
        pointMax = Point.End(self._builder)

        frameId = self._builder.CreateString("map")
        Header.Start(self._builder)
        Header.AddFrameId(self._builder, frameId)
        header = Header.End(self._builder)

        Boundingbox.Start(self._builder)
        Boundingbox.AddPointMin(self._builder, pointMin)
        Boundingbox.AddPointMax(self._builder, pointMax)
        #Boundingbox.AddHeader(self._builder, header)
        boundingbox = Boundingbox.End(self._builder)

        #Query.AddBoundingbox(self._builder, boundingbox)
        return boundingbox

    def gen_boundingbox2dlabeledstamped (self, boundingBoxes):

        projuuid_str = self.string_to_fbmsg(self._projectid)
        msguuid_str = self.string_to_fbmsg(self._msguuid)

        # build header
        Header.Start(self._builder)
        Header.AddUuidProject(self._builder, projuuid_str)
        Header.AddUuidMsgs(self._builder, msguuid_str)
        header = Header.End(self._builder)

        # a labels_bb array which will hold all the bbs
        label_bbs = []

        # create bounding box(es)
        for bb in boundingBoxes:
            x = self.gen_boundingbox(bb[0], bb[1])
            label_bbs.append(x)

        BoundingBoxes2DLabeledStamped.StartLabelsBbVector(self._builder, len(label_bbs))
        for bb in reversed(label_bbs):
            self._builder.PrependUOffsetTRelative(bb)
        self._builder.EndVector()

        # Start a new bounding box 2d labeled stamped
        BoundingBoxes2DLabeledStamped.Start(self._builder)

        BoundingBoxes2DLabeledStamped.AddHeader(self._builder, header)

        for label_bb in label_bbs:
            BoundingBoxes2DLabeledStamped.AddLabelsBb(self._builder, label_bb)

        return BoundingBoxes2DLabeledStamped.End(self._builder)

    def gen_timestamp(self, starttime, endtime):
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

        #Query.AddTimeinterval(self._builder, timeInterval)
        return timeInterval

    def gen_label(self, label):
        label = builder.CreateString("1")
        Query.StartLabelVector(builder, 1)
        builder.PrependUOffsetTRelative(label)
        labelMsg = builder.EndVector()

        #Query.AddLabel(builder, labelMsg)
        return labelMsg

    def run_query(self, **kwargs):
        projectuuidString = self._builder.CreateString(self._projectid)
        Query.StartProjectuuidVector(self._builder, 1)
        self._builder.PrependUOffsetTRelative(projectuuidString)
        projectuuidMsg = self._builder.EndVector()

        Query.Start(self._builder)

        Query.AddProjectuuid(self._builder, projectuuidMsg)
# Query.AddWithoutdata
        for key, value in kwargs.items():
            if key == "bb": Query.AddBoundingbox(self._builder, value)
            if key == "ti": Query.AddTimeinterval(self._builder, value)

        queryMsg = Query.End(self._builder)

        self._builder.Finish(queryMsg)
        buf = self._builder.Output()
        
        data = []
        sample = {}

        for responseBuf in self._grpc_stub.GetImage(bytes(buf)):
            print('[INFO] Receiving images . . .')
            response = Image.Image.GetRootAs(responseBuf)

            # this should not be inside the loop
            self._msguuid = response.Header().UuidMsgs().decode("utf-8")
            sample['uuid'] = self._msguuid
            sample['image'] = np.reshape(response.DataAsNumpy(), (response.Height(), response.Width(), 3))

            # Uncomment to visualize images
            # import matplotlib.pyplot as plt
            # plt.imshow(np.reshape(response.DataAsNumpy(), (response.Height(), response.Width(), 3)))
            # plt.show()
            # TODO why are the bounding boxes empty?????
            nbbs = response.LabelsBbLength()
            sample['boxes'] = nbbs
            for x in range( nbbs ):
                # print("uuidmsg: " + response.Header().UuidMsgs().decode("utf-8"))
                # print("first label: " + response.LabelsBb(0).LabelWithInstance().Label().decode("utf-8"))
                # print(
                #     "first BoundingBox (Xmin,Ymin,Xmax,Ymax): "
                #     + str(response.LabelsBb(0).BoundingBox().PointMin().X())
                #     + " "
                #     + str(response.LabelsBb(0).BoundingBox().PointMin().Y())
                #     + " "
                #     + str(response.LabelsBb(0).BoundingBox().PointMax().X())
                #     + " "
                #     + str(response.LabelsBb(0).BoundingBox().PointMax().Y())
                # )
                print(f"uuidmsg: {response.Header().UuidMsgs().decode('utf-8')}")
                print("first label: " + response.LabelsBb(0).BoundingBox2dLabeled(0).LabelWithInstance().Label().Label().decode("utf-8") 
                    + " ; confidence: " 
                    + str(response.LabelsBb(0).BoundingBox2dLabeled(0).LabelWithInstance().Label().Confidence())
                     )
                print(
                    "first bounding box (Xcenter,Ycenter,Xextent,Yextent): "
                    + str(response.LabelsBb(0).BoundingBox2dLabeled(0).BoundingBox().CenterPoint().X())
                    + " "
                    + str(response.LabelsBb(0).BoundingBox2dLabeled(0).BoundingBox().CenterPoint().Y())
                    + " "
                    + str(response.LabelsBb(0).BoundingBox2dLabeled(0).BoundingBox().SpatialExtent().X())
                    + " "
                    + str(response.LabelsBb(0).BoundingBox2dLabeled(0).BoundingBox().SpatialExtent().Y())
                    + "\n"
                )
            data.append(sample)
            sample={}
        return data

    def sendboundingbox(self, sample, bbs, labels, model_name):
        header = util_fb.createHeader(
            self._builder,
            projectUuid = self._projectid,
            msgUuid= sample['uuid']
        )

        boundingBoxes = util_fb.createBoundingBoxes2d(
            self._builder,
            [util_fb.createPoint2d(self._builder, bb[0][0], bb[0][1]) for bb in bbs],
            [util_fb.createPoint2d(self._builder, bb[1][0], bb[1][1]) for bb in bbs],
        )
        labelWithInstances = util_fb.createLabelsWithInstance(
            self._builder,
            [label for label in labels],
            [str(uuid.uuid4()) for _ in range(len(bbs))],
        )
        labelsBb = util_fb.createBoundingBoxes2dLabeled(self._builder, labelWithInstances, boundingBoxes)

        boundingBox2DLabeledWithCategory = util_fb.createBoundingBox2DLabeledWithCategory(
            self._builder, self._builder.CreateString(model_name), labelsBb
        )

        labelsBbVector = util_fb.createBoundingBox2dLabeledStamped(self._builder, header, [boundingBox2DLabeledWithCategory])
        self._builder.Finish(labelsBbVector)
        buf = self._builder.Output()

        msg = [bytes(buf)]

        send_channel, _, _, _ = self.secondary_channel()
        try:
            ret = send_channel.AddBoundingBoxes2dLabeled( iter(msg) )
        except Exception as e:
            print(e)   


    '''
    def sendboundingbox(self, bb):
        send_channel, _, _, _ = self.secondary_channel()

        self._builder.Finish(bb)
        buf = self._builder.Output()
        bufBytes = [ bytes(buf) ]

        ret = send_channel.AddBoundingBoxes2dLabeled( iter(bufBytes) )

        print("[bb service call]" + str(ret.decode()))
    '''

def main():
    schan = SEEREPChannel()
    ts = schan.gen_timestamp(1610549273, 1938549273)
    schan.run_query(ti=ts)

if __name__ == "__main__":
    main()

