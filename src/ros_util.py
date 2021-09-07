#!/usr/bin/env python3

# ROS imports
import rospy
from sensor_msgs.msg import Image
from PIL import Image as pil_image
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# Further imports
from cv_bridge import CvBridge, CvBridgeError
import yaml
import cv2
# from preprocess import image_adjust

def get_image_msg():
    param_file = rospy.get_param('client_parameter_file', '../ros_to_infer/parameter/client_parameter.yaml')
    rospy.Subscriber(param['ros_topic'], Image, image_callback, callback_args=1)
    rospy.spin()

def publish_inference():
    rospy.init_node('triton_client')
    param_file = rospy.get_param('client_parameter_file', '../ros_to_infer/parameter/client_parameter.yaml')

    global param
    with open(param_file) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
        detection = rospy.Publisher('detection', Image, queue_size=10)
        rospy.Subscriber(param['ros_topic'], Image, image_callback, callback_args=detection)
        rospy.spin()

def image_callback(msg, args):
    # t1 = time.time()
    bridge = CvBridge()
    msg_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    image = pil_image.fromarray(msg_frame)
    image = pil_image.resize((256, 256), pil_image.BILINEAR)

    msg_frame.header.stamp = rospy.Time.now()


    return image_data


if __name__ == '__main__':
    publish_inference()
