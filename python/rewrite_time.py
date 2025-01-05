#!/usr/bin/env python

import rosbag
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import math
import numpy as np
from tqdm import tqdm

# def rotate_pose(pose):

#     R = np.array([[0, -1, 0],
#                   [1,  0, 0],
#                   [0,  0, 1]])
#     original_xyz = np.array([pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z])
#     transformed_xyz = R.dot(original_xyz)

#     pose.pose.pose.position.x = transformed_xyz[0]
#     pose.pose.pose.position.y = transformed_xyz[1]
#     pose.pose.pose.position.z = transformed_xyz[2]

#     return pose

def rewrite_rosbag(input_bagfile, output_bagfile):
    with rosbag.Bag(output_bagfile, 'w') as outbag:

        for topic, msg, t in rosbag.Bag(input_bagfile).read_messages():
            first_t = t
            first_sec = msg.header.stamp.secs
            break


        for topic, msg, t in tqdm(rosbag.Bag(input_bagfile).read_messages()):
            msg.header.stamp.secs = msg.header.stamp.secs - first_sec + 3
            # if topic == 'S1/vio_odom':  # Replace with your actual topic
            #     transformed_msg = rotate_pose(msg)
            outbag.write(topic, msg, t - first_t)
            # else:
            #     outbag.write(topic, msg, t)

            # i += 1
            # if i == 500:
            #     break

if __name__ == '__main__':
    filename = "drive_seq5"
    input_bagfile = filename + '.bag'  # Replace with your input bag file
    output_bagfile = filename + '_sync.bag'  # Replace with your desired output bag file
    rewrite_rosbag(input_bagfile, output_bagfile)
