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

# capnp.add_import_hook([current_path + '/../src/capnp', current_path + '/ecal-common/src/capnp'])
capnp.add_import_hook([current_path + '/../src/capnp'])


from enum import Enum
import rosbag
from rospy import rostime
import signal

import yaml

# import imulist_capnp as eCALImuList
# import image_capnp as eCALImage
# import flow2d_capnp as eCALHFFlow
# import disparity_capnp as eCALDisaprity



from cv_bridge import CvBridge
import cv2 as cv
from utils import SyncedImageSubscriber, ImuSubscriber, VioSubscriber, DSCamera

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, Imu, CompressedImage

import tf2_ros

import tf


from ecal.core.subscriber import MessageSubscriber
# from pynput import keyboard


# from byte_subscriber import ByteSubscriber



# class RosOdometryPublisher:
#     def publish_tf(self, tf_msg):
#         if not self.no_tf_publisher:
#                 self.broadcaster.sendTransform(tf_msg)

#     def publish_static_tf(self, tf_msg):
#         # we probably should always publish tf transform
#         # if not self.no_tf_publisher:
#         #         self.static_broadcaster.sendTransform(tf_msg)
#         self.static_broadcaster.sendTransform(tf_msg)

#     def __init__(self, ros_tf_prefix: str, topic: str, use_monotonic: bool, no_tf_publisher: bool, print_debug: bool) -> None:
#         self.first_message = True
#         self.ros_odom_pub = rospy.Publisher(topic, Odometry, queue_size=10)
#         self.use_monotonic = use_monotonic
#         self.no_tf_publisher = no_tf_publisher
#         self.ros_tf_prefix = ros_tf_prefix + "/"
#         self.print_debug = print_debug

#         if self.print_debug:
#             print(f"ecal-ros bridge using monotonic = {use_monotonic}")
#             print(f"ecal-ros bridge publishing tf = {not no_tf_publisher}")
#             print(f"ecal-ros bridge publish topic = {topic}, with tf prefix {self.ros_tf_prefix}")

#         # static transforms
#         self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         self.broadcaster = tf2_ros.TransformBroadcaster()

#         if topic.endswith("_ned"):
#             self.isNED = True
#         else:
#             self.isNED = False

#         self.tf_msg_odom_ned = TransformStamped()
#         if self.use_monotonic:
#             self.tf_msg_odom_ned.header.stamp = rospy.Time.from_sec(time.monotonic())
#         else:
#             self.tf_msg_odom_ned.header.stamp = rospy.Time.now()
#         self.tf_msg_odom_ned.header.frame_id = self.ros_tf_prefix + "odom"
#         self.tf_msg_odom_ned.child_frame_id = self.ros_tf_prefix + "odom_ned"

#         self.tf_msg_odom_ned.transform.translation.x = 0
#         self.tf_msg_odom_ned.transform.translation.y = 0
#         self.tf_msg_odom_ned.transform.translation.z = 0

#         # R_ned_nwu = np.array ([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
#         T_nwu_ned = np.identity(4)
#         R_nwu_ned = np.array ([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
#         T_nwu_ned[:3, :3] = R_nwu_ned
#         quat = tf.transformations.quaternion_from_matrix(T_nwu_ned)

#         self.tf_msg_odom_ned.transform.rotation.x = quat[0]
#         self.tf_msg_odom_ned.transform.rotation.y = quat[1]
#         self.tf_msg_odom_ned.transform.rotation.z = quat[2]
#         self.tf_msg_odom_ned.transform.rotation.w = quat[3]

#         time.sleep(0.5)

#         self.publish_static_tf(self.tf_msg_odom_ned)
        

#         time.sleep(0.1)

#         self.tf_msg_base_link = TransformStamped()
#         self.tf_msg_base_link.header.stamp = self.tf_msg_odom_ned.header.stamp
#         self.tf_msg_base_link.transform = self.tf_msg_odom_ned.transform

#         self.tf_msg_base_link.header.frame_id = self.ros_tf_prefix + "base_link"
#         self.tf_msg_base_link.child_frame_id = self.ros_tf_prefix + "base_link_frd"

#         self.publish_static_tf(self.tf_msg_base_link)

#         self.tf_msg_odom_nwu = TransformStamped()
#         self.tf_msg_odom_nwu.header.stamp = self.tf_msg_odom_ned.header.stamp

#         self.tf_msg_odom_nwu.header.frame_id = self.ros_tf_prefix + "odom"
#         self.tf_msg_odom_nwu.child_frame_id = self.ros_tf_prefix + "odom_nwu"

#         self.tf_msg_odom_nwu.transform.translation.x = 0
#         self.tf_msg_odom_nwu.transform.translation.y = 0
#         self.tf_msg_odom_nwu.transform.translation.z = 0

#         self.tf_msg_odom_nwu.transform.rotation.x = 0
#         self.tf_msg_odom_nwu.transform.rotation.y = 0
#         self.tf_msg_odom_nwu.transform.rotation.z = 0
#         self.tf_msg_odom_nwu.transform.rotation.w = 1

#         self.publish_static_tf(self.tf_msg_odom_nwu)


#     def callback(self, topic_name, msg, time_ecal):

#         # need to remove the .decode() function within the Python API of ecal.core.subscriber ByteSubscriber
        
#         with eCALOdometry3d.Odometry3d.from_bytes(msg) as odometryMsg:
#             if odometryMsg.header.seq % 100 == 0:
#                 if self.use_monotonic:
#                     self.tf_msg_odom_ned.header.stamp = rospy.Time.from_sec(time.monotonic())
#                     self.tf_msg_base_link.header.stamp = rospy.Time.from_sec(time.monotonic())
#                     self.tf_msg_odom_nwu.header.stamp = rospy.Time.from_sec(time.monotonic())
#                 else:
#                     self.tf_msg_odom_ned.header.stamp = rospy.Time.now()
#                     self.tf_msg_base_link.header.stamp = rospy.Time.now()
#                     self.tf_msg_odom_nwu.header.stamp = rospy.Time.now()

#                 self.publish_static_tf(self.tf_msg_odom_ned)
#                 self.publish_static_tf(self.tf_msg_base_link)
#                 self.publish_static_tf(self.tf_msg_odom_nwu)

#             ros_msg = Odometry()
#             ros_msg.header.seq = odometryMsg.header.seq

#             if self.use_monotonic:
#                 ros_msg.header.stamp = rospy.Time.from_sec(odometryMsg.header.stamp / 1.0e9)
#             else:
#                 ros_msg.header.stamp = rospy.Time.now() #.from_sec(odometryMsg.header.stamp / 1.0e9)

#             if self.isNED:
#                 ros_msg.header.frame_id = self.ros_tf_prefix + "odom_ned"
#                 ros_msg.child_frame_id = self.ros_tf_prefix + "base_link_frd"
#             else:
#                 ros_msg.header.frame_id = self.ros_tf_prefix + "odom"
#                 ros_msg.child_frame_id = self.ros_tf_prefix + "base_link"

#             ros_msg.pose.pose.position.x = odometryMsg.pose.position.x
#             ros_msg.pose.pose.position.y = odometryMsg.pose.position.y
#             ros_msg.pose.pose.position.z = odometryMsg.pose.position.z

#             ros_msg.pose.pose.orientation.w = odometryMsg.pose.orientation.w
#             ros_msg.pose.pose.orientation.x = odometryMsg.pose.orientation.x
#             ros_msg.pose.pose.orientation.y = odometryMsg.pose.orientation.y
#             ros_msg.pose.pose.orientation.z = odometryMsg.pose.orientation.z

#             self.ros_odom_pub.publish(ros_msg)

#             # publish

#             tf_msg = TransformStamped()
#             tf_msg.header.stamp = ros_msg.header.stamp

#             if self.isNED:
#                 tf_msg.header.frame_id = self.ros_tf_prefix + "odom_ned"
#                 tf_msg.child_frame_id = self.ros_tf_prefix + "base_link_frd"
#             else:
#                 tf_msg.header.frame_id = self.ros_tf_prefix + "odom"
#                 tf_msg.child_frame_id = self.ros_tf_prefix + "base_link"

#             tf_msg.transform.translation.x = odometryMsg.pose.position.x
#             tf_msg.transform.translation.y = odometryMsg.pose.position.y
#             tf_msg.transform.translation.z = odometryMsg.pose.position.z

#             tf_msg.transform.rotation = ros_msg.pose.pose.orientation

#             self.publish_tf(tf_msg)

# class RosImuListPublisher:
#     def __init__(self, pub_topic: str, print_debug: bool) -> None:
#         self.ros_imu_pub = rospy.Publisher(pub_topic, Imu, queue_size=3000)
#         self.print_debug = print_debug


class RosImuPublisher:
    def __init__(self, pub_topic: str, print_debug: bool) -> None:
        self.ros_imu_pub = rospy.Publisher(pub_topic, Imu, queue_size=3000)
        self.print_debug = print_debug

    def ecal2ros_imu(self, ecal_imuMsg):
        ros_msg_imu = Imu()
        stamp = rostime.Time(int(ecal_imuMsg.header.stamp // 1e9), ecal_imuMsg.header.stamp % 1e9)

        ros_msg_imu.header.stamp = stamp
        ros_msg_imu.header.frame_id = "map"
        ros_msg_imu.angular_velocity.x = ecal_imuMsg.angularVelocity.x
        ros_msg_imu.angular_velocity.y = ecal_imuMsg.angularVelocity.y
        ros_msg_imu.angular_velocity.z = ecal_imuMsg.angularVelocity.z

        ros_msg_imu.linear_acceleration.x = ecal_imuMsg.linearAcceleration.x
        ros_msg_imu.linear_acceleration.y = ecal_imuMsg.linearAcceleration.y
        ros_msg_imu.linear_acceleration.z = ecal_imuMsg.linearAcceleration.z
        ros_msg_imu.header.seq = ecal_imuMsg.header.seq
        return ros_msg_imu, stamp

    def callback(self, ecal_imuMsg):
        ros_msg_imu, stamp = self.ecal2ros_imu(ecal_imuMsg)
        self.ros_imu_pub.publish(ros_msg_imu)

        return ros_msg_imu, stamp


class RosPinholeImagePublisher:
    def __init__(self, pub_topic: str, print_debug: bool) -> None:
        self.ros_image_pub = rospy.Publisher(pub_topic, Image, queue_size=10)
        # self.ros_image_pub = rospy.Publisher(pub_topic, CompressedImage, queue_size=100)
        self.bridge = CvBridge()

    def ecal2ros_image(self, ecal_imageMsg):
        stamp = rostime.Time(int(ecal_imageMsg.header.stamp // 1e9), ecal_imageMsg.header.stamp % 1e9)        
        decode_img = cv.imdecode(np.frombuffer(ecal_imageMsg.data, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
        ros_msg_image = self.bridge.cv2_to_imgmsg(decode_img, encoding="mono8")
        # ros_msg_image = self.bridge.cv2_to_compressed_imgmsg(decode_img, dst_format="jpeg")

        ros_msg_image.header.seq = ecal_imageMsg.header.seq
        ros_msg_image.header.stamp = stamp

        return ros_msg_image, stamp

    def callback(self, ecal_imageMsg):
        ros_msg_image, stamp = self.ecal2ros_image(ecal_imageMsg)
        self.ros_image_pub.publish(ros_msg_image)

        return ros_msg_image, stamp

class RosFisheyeImagePublisher:
    def __init__(self, pub_topic: str, print_debug: bool) -> None:
        self.ros_image_pub = rospy.Publisher(pub_topic, Image, queue_size=10)
        # self.ros_image_pub = rospy.Publisher(pub_topic, CompressedImage, queue_size=100)
        self.bridge = CvBridge()
        self.cam = None

    def ecal2ros_image(self, ecal_imageMsg):
        stamp = rostime.Time(int(ecal_imageMsg.header.stamp // 1e9), ecal_imageMsg.header.stamp % 1e9)        
        decode_img = cv.imdecode(np.frombuffer(ecal_imageMsg.data, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
        
        if self.cam == None:
            self.cam = DSCamera(ecal_imageMsg.width,                                
                                ecal_imageMsg.height,
                                ecal_imageMsg.intrinsic.ds.pinhole.fx,
                                ecal_imageMsg.intrinsic.ds.pinhole.fy,
                                ecal_imageMsg.intrinsic.ds.pinhole.cx,
                                ecal_imageMsg.intrinsic.ds.pinhole.cy,
                                ecal_imageMsg.intrinsic.ds.xi,
                                ecal_imageMsg.intrinsic.ds.alpha,
                                fov=180)

        decode_img = self.cam.to_perspective(decode_img)
        ros_msg_image = self.bridge.cv2_to_imgmsg(decode_img, encoding="mono8")
        ros_msg_image.header.seq = ecal_imageMsg.header.seq
        ros_msg_image.header.stamp = stamp

        return ros_msg_image, stamp

    def callback(self, ecal_imageMsg):
        ros_msg_image, stamp = self.ecal2ros_image(ecal_imageMsg)
        self.ros_image_pub.publish(ros_msg_image)

        return ros_msg_image, stamp


class RosVioPublisher:
    def __init__(self, pub_topic: str, print_debug: bool) -> None:
        self.ros_vio_pub = rospy.Publisher(pub_topic, Odometry, queue_size=3000)
        self.print_debug = print_debug

    def ecal2ros_vio(self, ecal_vioMsg):
        ros_msg_vio = Odometry()
        stamp = rostime.Time(int(ecal_vioMsg.header.stamp // 1e9), ecal_vioMsg.header.stamp % 1e9)
        
        ros_msg_vio.header.stamp = stamp
        ros_msg_vio.header.seq = ecal_vioMsg.header.seq
        ros_msg_vio.header.frame_id = "map"
        ros_msg_vio.child_frame_id = ""

        ros_msg_vio.pose.pose.position.x = ecal_vioMsg.pose.position.x
        ros_msg_vio.pose.pose.position.y = ecal_vioMsg.pose.position.y
        ros_msg_vio.pose.pose.position.z = ecal_vioMsg.pose.position.z
        ros_msg_vio.pose.pose.orientation.x = ecal_vioMsg.pose.orientation.x
        ros_msg_vio.pose.pose.orientation.y = ecal_vioMsg.pose.orientation.y
        ros_msg_vio.pose.pose.orientation.z = ecal_vioMsg.pose.orientation.z
        ros_msg_vio.pose.pose.orientation.w = ecal_vioMsg.pose.orientation.w

        ros_msg_vio.twist.twist.linear.x = ecal_vioMsg.twist.linear.x
        ros_msg_vio.twist.twist.linear.y = ecal_vioMsg.twist.linear.y
        ros_msg_vio.twist.twist.linear.z = ecal_vioMsg.twist.linear.z
        ros_msg_vio.twist.twist.angular.x = ecal_vioMsg.twist.angular.x
        ros_msg_vio.twist.twist.angular.y = ecal_vioMsg.twist.angular.y
        ros_msg_vio.twist.twist.angular.z = ecal_vioMsg.twist.angular.z

        # Not necessary for now
        # ros_msg_vio.pose.covariance = ecal_vioMsg.poseCovariance
        # ros_msg_vio.twist.covariance = ecal_vioMsg.twistCovariance

        return ros_msg_vio, stamp

    def callback(self, ecal_vioMsg):
        ros_msg_vio, stamp = self.ecal2ros_vio(ecal_vioMsg)
        self.ros_vio_pub.publish(ros_msg_vio)
        # print(dir(ecal_vioMsg))
        # with ecal_vioMsg.Odometry3d.from_bytes(msg) as odometryMsg:


        return ros_msg_vio, stamp



class Ecal2ROS:

    def __init__(self, args):
        self.run = True
        self.recording = False
        self.bag = None      
        self.device = args.device
        self.print_debug = args.print_debug
        self.configs_dir = args.configs_dir
        self.bag_dir = args.bag_dir
        self.bag_name = args.bag_name
        self.recording = args.record

        # Setup ecal subscribers and ros publishers
        self.setup_topics()

        # Start ecal2ros threads
        self.start_threads()


    def setup_topics(self):
        device_topics = self.get_topics()

        # ROS Publishers
        self.image_publishers = {}
        image_topics = []
        for topic in device_topics:
            message_type = device_topics[topic]
            if message_type == "imu":
                self.last_imu_seq = -1
                self.imu_publisher = RosImuPublisher(topic, self.print_debug)
                self.imu_sub = ImuSubscriber(topic)

            # elif message_type == "imulist":
            #     self.imu_publisher = RosImuListPublisher(topic, self.print_debug)
            #     self.imu_sub = ImuSubscriber(topic)
            
            elif message_type == "viostate":
                pass

            elif message_type == "odometry3d":
                self.last_vio_seq = -1
                self.vio_publisher = RosVioPublisher(topic, self.print_debug)
                self.vio_sub = VioSubscriber(topic)

            elif message_type == "hfopticalflow":
                pass

            elif message_type == "disparity":
                pass

            elif message_type == "image":
                if "camd" in topic:
                    self.image_publishers[topic] = RosFisheyeImagePublisher(topic, self.print_debug)
                else:
                    self.image_publishers[topic] = RosPinholeImagePublisher(topic, self.print_debug)
                image_topics.append(topic)

        # Ecal Subscribers
        self.image_sub = SyncedImageSubscriber(image_topics)


    def get_topics(self):
        with open(self.configs_dir, 'r') as yaml_file:
            topics = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return topics[self.device]


    def start_threads(self):
        self.thread_imu = threading.Thread(target=self.continuous_thread_imu)
        self.thread_image = threading.Thread(target=self.continuous_thread_image)
        self.thread_vio = threading.Thread(target=self.continuous_thread_vio)

        self.imu_sub.rolling = True
        self.image_sub.rolling = True
        self.vio_sub.rolling = True

        self.thread_imu.start()
        self.thread_image.start()
        self.thread_vio.start()

        if self.recording:
            self.start_record()


    def start_record(self):
        print("Start recording topics.")
        if self.bag is None:
            if self.bag_name == None:
                self.bag_name = time.strftime('%H%M%S', time.localtime()) + "_ecal2ros.bag"
            else:
                self.bag_name += ".bag"
            bag_path = self.bag_dir.joinpath(self.bag_name)
            self.bag = rosbag.Bag(bag_path.resolve(), mode='w', compression=rosbag.Compression.NONE)
            self.bag_lock = threading.Lock()

        if self.recording:
            print("continuous recording already started")
            return
        else:
            self.recording = True


    def stop_record(self):
        self.recording = False

        # if self.thread_image is not None:
        #     self.thread_image.join(3)
        #     self.thread_imu.join(3)
        #     print("stop record")

        if self.bag is not None:
            print("bag file closing")
            self.bag.close()
            self.bag = None
            print("Recorder stopped")
        else:
            print("No recording started")


    def continuous_thread_image(self):
        print("Start streaming image...")
        while self.run:
            images = self.image_sub.pop_sync_queue()

            if images is not None:
                for topic in images:
                    imageMsg = images[topic]


                    # To add in debug
                    # imagemsg_print = imageMsg.to_dict()
                    # imagemsg_print.pop("data")
                    # print("=================")
                    # print(imagemsg_print)
                    # print("=================")

                    ros_msg_image, stamp = self.image_publishers[topic].callback(imageMsg)

                    if self.recording:
                        with self.bag_lock:
                            # this may block for a while..
                            self.bag.write(topic, ros_msg_image, stamp)

        print("continuous_thread_image stopping")

    def continuous_thread_imu(self):
        print("Start streaming imu...")
        while self.run:
            imu_item = self.imu_sub.pop_queue()
            if self.last_imu_seq >= 0:
                if self.last_imu_seq + 1 != imu_item.header.seq:
                    print("imu jump detected for bag writing", self.last_imu_seq, imu_item.header.seq, "Missing", self.last_imu_seq + 1)
                    pass
            self.last_imu_seq = imu_item.header.seq
            ros_msg_imu, stamp = self.imu_publisher.callback(imu_item)

            if self.recording:
                with self.bag_lock:
                    # this may block for a while..
                    self.bag.write(self.imu_sub.topic, ros_msg_imu, stamp)

        print("continuous_thread_imu stopping")

    def continuous_thread_vio(self):
        print("Start streaming vio...")
        while self.run:
            vio_item = self.vio_sub.pop_queue()
            if self.last_vio_seq >= 0:
                if self.last_vio_seq + 1 != vio_item.header.seq:
                    print("vio jump detected for bag writing", self.last_vio_seq, vio_item.header.seq, "Missing", self.last_vio_seq + 1)
                    pass
            self.last_vio_seq = vio_item.header.seq
            ros_msg_vio, stamp = self.vio_publisher.callback(vio_item)

            if self.recording:
                with self.bag_lock:
                    # this may block for a while..
                    self.bag.write(self.vio_sub.topic, ros_msg_vio, stamp)

        print("continuous_thread_vio stopping")


# def on_release(key, recorder):
#     if key == keyboard.Key.space:
#         print("Spacebar pressed. Recording starting")
#         return recorder.start_record()
        
#     elif key.char == 'q':
#         print(" pressed. Recording stopping")
#         return recorder.stop_record()


def main():
    # print eCAL version and date
    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='dp180_master')
    parser.add_argument('--bag_name', type=str, default=None)
    parser.add_argument('--bag_dir', type=str, default=pathlib.Path("~/vilota/rosbags/"))
    parser.add_argument('--configs_dir', type=str, default=pathlib.Path("devices.yaml"))
    # parser.add_argument('--ros_tf_prefix', type=str, default='S1')
    # parser.add_argument('--no_tf_publisher', action="store_true")
    parser.add_argument('--record', action="store_true")
    parser.add_argument('--monotonic_time', action="store_true")
    parser.add_argument('--print_debug', action="store_true")
    args = parser.parse_known_args()[0]

    # Initialize eCAL API
    ecal_core.initialize(sys.argv, "ecal2ros_publisher")

    # set process state
    ecal_core.set_process_state(1, 1, "I feel good")

    # Initialize publisher node
    rospy.init_node("ecal2ros_publisher")

    def handler(signum, frame):
        print("ctrl-c is pressed")
        recorder.stop_record()
        exit(1)
    signal.signal(signal.SIGINT, handler)
    time.sleep(1)


    recorder = Ecal2ROS(args)

    # # Keyboard Listener
    # listener = keyboard.Listener(on_release=lambda event: on_release(event, self))
    # listener.start()

    rospy.spin()

    # finalize eCAL API
    ecal_core.finalize()

if __name__ == "__main__":
    main()