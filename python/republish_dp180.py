#!/usr/bin/env python3

import sys
import time
import threading
import argparse

import capnp
import numpy as np

import ecal.core.core as ecal_core

import pathlib

current_path = str(pathlib.Path(__file__).parent.resolve())

print("working in path " + current_path)

capnp.add_import_hook([current_path + '/../src/capnp', current_path + '/ecal-common/src/capnp'])


import odometry3d_capnp as eCALOdometry3d
import disparity_capnp as eCALDisaprity
import image_capnp as eCALImage
import imu_capnp as eCALImu
from cv_bridge import CvBridge
import cv2 as cv
from utils import SyncedImageSubscriber, ImuSubscriber, image_resize

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image

import tf2_ros

import tf


from ecal.core.subscriber import MessageSubscriber

class CapnpSubscriber(MessageSubscriber):
  """Specialized publisher subscribes to raw bytes
  """
  def __init__(self, type, name, typeclass=None):
    self.topic_type = "capnp:" + type
    super(CapnpSubscriber, self).__init__(name, self.topic_type)
    self.callback = None
    self.typeclass = typeclass

  def receive(self, timeout=0):
    """ receive subscriber content with timeout

    :param timeout: receive timeout in ms

    """
    ret, msg, time = self.c_subscriber.receive(timeout)
    return ret, msg, time

  def set_callback(self, callback):
    """ set callback function for incoming messages

    :param callback: python callback function (f(topic_name, msg, time))

    """
    self.callback = callback
    self.c_subscriber.set_callback(self._on_receive)

  def rem_callback(self, callback):
    """ remove callback function for incoming messages

    :param callback: python callback function (f(topic_type, topic_name, msg, time))

    """
    self.c_subscriber.rem_callback(self._on_receive)
    self.callback = None

  def _on_receive(self, topic_name, msg, time):
    if self.typeclass is None:
      self.callback(self.topic_type, topic_name, msg, time)
    else:
      with self.typeclass.from_bytes(msg) as msg:
        self.callback(self.topic_type, topic_name, msg, time)

class ByteSubscriber(MessageSubscriber):
  """Specialized publisher subscribes to raw bytes
  """
  def __init__(self, name):
    topic_type = "base:byte"
    super(ByteSubscriber, self).__init__(name, topic_type)
    self.callback = None

  def receive(self, timeout=0):
    """ receive subscriber content with timeout

    :param timeout: receive timeout in ms

    """
    ret, msg, time = self.c_subscriber.receive(timeout)
    return ret, msg, time

  def set_callback(self, callback):
    """ set callback function for incoming messages

    :param callback: python callback function (f(topic_name, msg, time))

    """
    self.callback = callback
    self.c_subscriber.set_callback(self._on_receive)

  def rem_callback(self, callback):
    """ remove callback function for incoming messages

    :param callback: python callback function (f(topic_name, msg, time))

    """
    self.c_subscriber.rem_callback(self._on_receive)
    self.callback = None

  def _on_receive(self, topic_name, msg, time):
    self.callback(topic_name, msg, time)    


class RosOdometryPublisher:

    def publish_tf(self, tf_msg):
        if not self.no_tf_publisher:
                self.broadcaster.sendTransform(tf_msg)

    def publish_static_tf(self, tf_msg):
        # we probably should always publish tf transform
        # if not self.no_tf_publisher:
        #         self.static_broadcaster.sendTransform(tf_msg)
        self.static_broadcaster.sendTransform(tf_msg)

    def __init__(self, ros_tf_prefix: str, topic: str, use_monotonic: bool, no_tf_publisher: bool, print_debug: bool) -> None:
        self.first_message = True
        self.ros_odom_pub = rospy.Publisher(topic, Odometry, queue_size=10)
        self.use_monotonic = use_monotonic
        self.no_tf_publisher = no_tf_publisher
        self.ros_tf_prefix = ros_tf_prefix + "/"
        self.print_debug = print_debug

        if self.print_debug:
            print(f"ecal-ros bridge using monotonic = {use_monotonic}")
            print(f"ecal-ros bridge publishing tf = {not no_tf_publisher}")
            print(f"ecal-ros bridge publish topic = {topic}, with tf prefix {self.ros_tf_prefix}")

        # static transforms
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.broadcaster = tf2_ros.TransformBroadcaster()

        if topic.endswith("_ned"):
            self.isNED = True
        else:
            self.isNED = False

        self.tf_msg_odom_ned = TransformStamped()
        if self.use_monotonic:
            self.tf_msg_odom_ned.header.stamp = rospy.Time.from_sec(time.monotonic())
        else:
            self.tf_msg_odom_ned.header.stamp = rospy.Time.now()
        self.tf_msg_odom_ned.header.frame_id = self.ros_tf_prefix + "odom"
        self.tf_msg_odom_ned.child_frame_id = self.ros_tf_prefix + "odom_ned"

        self.tf_msg_odom_ned.transform.translation.x = 0
        self.tf_msg_odom_ned.transform.translation.y = 0
        self.tf_msg_odom_ned.transform.translation.z = 0

        # R_ned_nwu = np.array ([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T_nwu_ned = np.identity(4)
        R_nwu_ned = np.array ([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T_nwu_ned[:3, :3] = R_nwu_ned
        quat = tf.transformations.quaternion_from_matrix(T_nwu_ned)

        self.tf_msg_odom_ned.transform.rotation.x = quat[0]
        self.tf_msg_odom_ned.transform.rotation.y = quat[1]
        self.tf_msg_odom_ned.transform.rotation.z = quat[2]
        self.tf_msg_odom_ned.transform.rotation.w = quat[3]

        time.sleep(0.5)

        self.publish_static_tf(self.tf_msg_odom_ned)
        

        time.sleep(0.1)

        self.tf_msg_base_link = TransformStamped()
        self.tf_msg_base_link.header.stamp = self.tf_msg_odom_ned.header.stamp
        self.tf_msg_base_link.transform = self.tf_msg_odom_ned.transform

        self.tf_msg_base_link.header.frame_id = self.ros_tf_prefix + "base_link"
        self.tf_msg_base_link.child_frame_id = self.ros_tf_prefix + "base_link_frd"

        self.publish_static_tf(self.tf_msg_base_link)

        self.tf_msg_odom_nwu = TransformStamped()
        self.tf_msg_odom_nwu.header.stamp = self.tf_msg_odom_ned.header.stamp

        self.tf_msg_odom_nwu.header.frame_id = self.ros_tf_prefix + "odom"
        self.tf_msg_odom_nwu.child_frame_id = self.ros_tf_prefix + "odom_nwu"

        self.tf_msg_odom_nwu.transform.translation.x = 0
        self.tf_msg_odom_nwu.transform.translation.y = 0
        self.tf_msg_odom_nwu.transform.translation.z = 0

        self.tf_msg_odom_nwu.transform.rotation.x = 0
        self.tf_msg_odom_nwu.transform.rotation.y = 0
        self.tf_msg_odom_nwu.transform.rotation.z = 0
        self.tf_msg_odom_nwu.transform.rotation.w = 1

        self.publish_static_tf(self.tf_msg_odom_nwu)


    def callback(self, topic_name, msg, time_ecal):

        # need to remove the .decode() function within the Python API of ecal.core.subscriber ByteSubscriber
        
        with eCALOdometry3d.Odometry3d.from_bytes(msg) as odometryMsg:
            if self.first_message:
                print(f"bodyFrame = {odometryMsg.bodyFrame}")
                print(f"referenceFrame = {odometryMsg.referenceFrame}")
                print(f"velocityFrame = {odometryMsg.velocityFrame}")
                self.first_message = False

            if odometryMsg.header.seq % 100 == 0:
                if self.print_debug:
                    print(f"seq = {odometryMsg.header.seq}")
                    print(f"latency device = {odometryMsg.header.latencyDevice / 1e6} ms")
                    print(f"latency host = {odometryMsg.header.latencyHost / 1e6} ms")
                    print(f"position = {odometryMsg.pose.position.x}, {odometryMsg.pose.position.y}, {odometryMsg.pose.position.z}")
                    print(f"orientation = {odometryMsg.pose.orientation.w}, {odometryMsg.pose.orientation.x}, {odometryMsg.pose.orientation.y}, {odometryMsg.pose.orientation.z}")
                    
                if self.use_monotonic:
                    self.tf_msg_odom_ned.header.stamp = rospy.Time.from_sec(time.monotonic())
                    self.tf_msg_base_link.header.stamp = rospy.Time.from_sec(time.monotonic())
                    self.tf_msg_odom_nwu.header.stamp = rospy.Time.from_sec(time.monotonic())
                else:
                    self.tf_msg_odom_ned.header.stamp = rospy.Time.now()
                    self.tf_msg_base_link.header.stamp = rospy.Time.now()
                    self.tf_msg_odom_nwu.header.stamp = rospy.Time.now()

                self.publish_static_tf(self.tf_msg_odom_ned)
                self.publish_static_tf(self.tf_msg_base_link)
                self.publish_static_tf(self.tf_msg_odom_nwu)

            ros_msg = Odometry()
            ros_msg.header.seq = odometryMsg.header.seq

            if self.use_monotonic:
                ros_msg.header.stamp = rospy.Time.from_sec(odometryMsg.header.stamp / 1.0e9)
            else:
                ros_msg.header.stamp = rospy.Time.now() #.from_sec(odometryMsg.header.stamp / 1.0e9)

            if self.isNED:
                ros_msg.header.frame_id = self.ros_tf_prefix + "odom_ned"
                ros_msg.child_frame_id = self.ros_tf_prefix + "base_link_frd"
            else:
                ros_msg.header.frame_id = self.ros_tf_prefix + "odom"
                ros_msg.child_frame_id = self.ros_tf_prefix + "base_link"

            ros_msg.pose.pose.position.x = odometryMsg.pose.position.x
            ros_msg.pose.pose.position.y = odometryMsg.pose.position.y
            ros_msg.pose.pose.position.z = odometryMsg.pose.position.z

            ros_msg.pose.pose.orientation.w = odometryMsg.pose.orientation.w
            ros_msg.pose.pose.orientation.x = odometryMsg.pose.orientation.x
            ros_msg.pose.pose.orientation.y = odometryMsg.pose.orientation.y
            ros_msg.pose.pose.orientation.z = odometryMsg.pose.orientation.z

            self.ros_odom_pub.publish(ros_msg)

            # publish

            tf_msg = TransformStamped()
            tf_msg.header.stamp = ros_msg.header.stamp

            if self.isNED:
                tf_msg.header.frame_id = self.ros_tf_prefix + "odom_ned"
                tf_msg.child_frame_id = self.ros_tf_prefix + "base_link_frd"
            else:
                tf_msg.header.frame_id = self.ros_tf_prefix + "odom"
                tf_msg.child_frame_id = self.ros_tf_prefix + "base_link"

            tf_msg.transform.translation.x = odometryMsg.pose.position.x
            tf_msg.transform.translation.y = odometryMsg.pose.position.y
            tf_msg.transform.translation.z = odometryMsg.pose.position.z

            tf_msg.transform.rotation = ros_msg.pose.pose.orientation

            self.publish_tf(tf_msg)


class RosImagePublisher:
    def __init__(self, pub_topic: str, use_monotonic: bool, print_debug: bool) -> None:
        self.ros_img_pub = rospy.Publisher(pub_topic, Image, queue_size=10)
        self.use_monotonic = use_monotonic
        self.print_debug = print_debug
        self.bridge = CvBridge()

    def callback(self, type, topic_name, msg, ts):
        with eCALImage.Image.from_bytes(msg) as imageMsg:
            if self.print_debug:                
                
                print(
                    f"seq = {imageMsg.header.seq}, stamp = {imageMsg.header.stamp}, with {len(msg)} bytes, encoding = {imageMsg.encoding}")
                print(f"latency device = {imageMsg.header.latencyDevice / 1e6} ms")
                print(f"latency host = {imageMsg.header.latencyHost / 1e6} ms")
                print(f"width = {imageMsg.width}, height = {imageMsg.height}")
                print(f"exposure = {imageMsg.exposureUSec}, gain = {imageMsg.gain}")
                print(f"intrinsic = {imageMsg.intrinsic}")
                print(f"extrinsic = {imageMsg.extrinsic}")
                print(f"instant w = {imageMsg.motionMeta.instantaneousAngularVelocity}")
                print(f"average w = {imageMsg.motionMeta.averageAngularVelocity}")
            assert False
            decode_img = cv.imdecode(np.frombuffer(imageMsg.data, dtype=np.uint8), -1)
            ros_msg = self.bridge.cv2_to_imgmsg(decode_img, encoding="passthrough")

            if self.use_monotonic:
                ros_msg.header.stamp = rospy.Time.from_sec(time.monotonic())
            else:
                ros_msg.header.stamp = rospy.Time.now()

            self.ros_img_pub.publish(ros_msg)

class RosImuPublisher:
    def __init__(self, pub_topic: str, use_monotonic: bool, print_debug: bool) -> None:
        self.ros_imu_pub = rospy.Publisher(pub_topic, Image, queue_size=10)
        self.use_monotonic = use_monotonic
        self.print_debug = print_debug


    def callback(self, type, topic_name, msg, ts):
        with eCALImage.Image.from_bytes(msg) as imageMsg:
            if self.print_debug:
                print(
                    f"seq = {imageMsg.header.seq}, stamp = {imageMsg.header.stamp}, with {len(msg)} bytes, encoding = {imageMsg.encoding}")
                print(f"latency device = {imageMsg.header.latencyDevice / 1e6} ms")
                print(f"latency host = {imageMsg.header.latencyHost / 1e6} ms")
                print(f"width = {imageMsg.width}, height = {imageMsg.height}")
                print(f"exposure = {imageMsg.exposureUSec}, gain = {imageMsg.gain}")
                print(f"intrinsic = {imageMsg.intrinsic}")
                print(f"extrinsic = {imageMsg.extrinsic}")
                print(f"instant w = {imageMsg.motionMeta.instantaneousAngularVelocity}")
                print(f"average w = {imageMsg.motionMeta.averageAngularVelocity}")

            decode_img = cv.imdecode(np.frombuffer(imageMsg.data, dtype=np.uint8), -1)
            ros_msg = self.bridge.cv2_to_imgmsg(decode_img, encoding="passthrough")

            if self.use_monotonic:
                ros_msg.header.stamp = rospy.Time.from_sec(time.monotonic())
            else:
                ros_msg.header.stamp = rospy.Time.now()

            self.ros_img_pub.publish(ros_msg)





def device_topics(args, topic):
    devices = {
                "dp180":    {
                            "S0/stereo2_r":             0,
                            "S0/vio_state":             0,
                            "S0/vio_odom_ned":          0,
                            "S0/vio_odom":              [ByteSubscriber("S0/vio_odom"), RosOdometryPublisher(args.ros_tf_prefix, "/basalt/odom_ned", args.monotonic_time, args.no_tf_publisher, args.print_debug)],
                            "S0/stereo1_l/hfflow":      0,
                            "S0/stereo1_l/disparity":   0,
                            "S0/stereo1_l":             [CapnpSubscriber("Image", "S0/stereo1_l"), RosImagePublisher("S0/stereo1_l", args.monotonic_time, args.print_debug)],
                            "S0/stereo2_r/hfflow":      0,
                            "S0/stereo2_r/disparity":   0,
                            "S0/stereo2_r":             [CapnpSubscriber("Image", "S0/stereo2_r"), RosImagePublisher("S0/stereo2_r", args.monotonic_time, args.print_debug)],
                            "S0/imu_list":              0,
                            "S0/imu":                   0,
                            "S0/camd/hfflow":           0,
                            "S0/camd":                  [CapnpSubscriber("Image", "S0/camd"), RosImagePublisher("S0/camd", args.monotonic_time, args.print_debug)],
                            }
               }
    return devices[args.device][topic]

def main():  

    # print eCAL version and date
    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))


    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='dp180')
    parser.add_argument('--ros_tf_prefix', type=str, default='S0')
    parser.add_argument('--monotonic_time', action="store_true")
    parser.add_argument('--no_tf_publisher', action="store_true")
    parser.add_argument('--print_debug', action="store_true")
    args = parser.parse_known_args()[0]
    
    # initialize eCAL API
    ecal_core.initialize(sys.argv, "test_odometry_sub")
    
    # set process state
    ecal_core.set_process_state(1, 1, "I feel good")


    topics_to_publish = [
                            # "S0/vio_odom",
                            "S0/camd",
                            # "S0/stereo2_r",
                        ]


    image_topics = ["S0/camd"]
    image_types = ["Image"]
    image_typeclasses = [eCALImage.Image]

    image_sub = SyncedImageSubscriber(image_types, image_topics, image_typeclasses)


    rospy.init_node("mcap2ros_publisher")

    for i, topic in enumerate(topics_to_publish):

        topic_nodes = device_topics(args, topic)

        sub = topic_nodes[0]
        pub = topic_nodes[1]
        sub.set_callback(pub.callback)


    # idle main thread
    # while ecal_core.ok():
    #     time.sleep(0.1)
    rospy.spin()

    # finalize eCAL API
    ecal_core.finalize()

if __name__ == "__main__":
    main()