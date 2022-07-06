
# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2
import numpy as np

import rosbag
from sensor_msgs.msg import PointCloud2
import ros_numpy

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("pc_topic", help="pointcloud topic.")

    args = parser.parse_args()

    print("Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.pc_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.pc_topic]):
        # pc = np.reshape(np.frombuffer(msg.data, dtype=np.int8), (-1, 4))
        pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        np.save(os.path.join(args.output_dir, "%06i" % count), pc)
        print("Wrote pointcloud %i" % count)

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()