import capnp
import numpy as np

import ecal.core.core as ecal_core
from capnp_subscriber import CapnpSubscriber

capnp.add_import_hook(['../src/capnp'])

import imu_capnp as eCALImu
import image_capnp as eCALImage
import odometry3d_capnp as eCALOdometry3d
# import viostate_capnp as eCALViostate

import time


import queue
from threading import Lock



from typing import Dict, Tuple

import cv2 as cv
import numpy as np


class ImuSubscriber:
    def __init__(self, topic):
        print(f"subscribing to imu topic {topic}")
        sub = self.subscriber = CapnpSubscriber("Imu", topic)
        sub.set_callback(self.callback)

        # self.lock = Lock()

        # self.warn_drop = False
        self.rolling = False

        self.m_queue = queue.Queue(2000)
        self.latest = None

        self.topic = topic

    # def queue_clear(self):
    #     with self.lock:
    #         self.m_queue = queue.Queue(200)

    def queue_update(self):

        if self.m_queue.full():
            if self.warn_drop is True:
                    print("imu queue is not processed in time")
            self.m_queue.get()
                

    def callback(self, topic_type, topic_name, msg, ts):
        with eCALImu.Imu.from_bytes(msg) as imuMsg:

            self.latest = imuMsg

            if self.rolling:
                try:
                    self.m_queue.put(imuMsg, block=False)
                except queue.Full:
                    print("imu queue full")
                    # print(self.m_queue.qsize())
                    self.m_queue.get()
                    self.m_queue.put(imuMsg)

            # self.queue_update()

    def pop_latest(self):

        # with self.lock:
        if self.latest == None:
            return {}
        else:
            return self.latest

    def pop_queue(self):
        # imu is quite frequent, it is ok to be blocked

        # if self.m_queue.qsize() > 10:
        #     print(self.m_queue.qsize())
        return self.m_queue.get()



class SyncedImageSubscriber:
    def __init__(self, 
                #  types, 
                 topics, 
                #  typeclasses=None, 
                 enforce_sync=True):

        self.subscribers = {}
        # store individual incoming images
        self.queues = {}
        # store properly synced images
        self.synced_queue = queue.Queue(150)
        self.enforce_sync = enforce_sync
        self.callbacks = []
        
        self.latest = None
        # assert len(types) == len(topics)
        # assert typeclasses is None or len(typeclasses) == len(topics)
        self.size = len(topics)

        self.rolling = False

        self.assemble = {}
        self.assemble_index = -1

        self.lock = Lock()

        for i in range(len(topics)):
            print(f"subscribing to topic {topics[i]}")
            # print(f"subscribing to {types[i]} topic {topics[i]}")
            # typeclass = typeclasses[i] if typeclasses is not None else None
            
            sub = self.subscribers[topics[i]] = CapnpSubscriber("Image", topics[i], eCALImage.Image)
            sub.set_callback(self.callback)

            self.queues[topics[i]] = queue.Queue(10)

    def queue_update(self):
        for queueName in self.queues:
            m_queue = self.queues[queueName]

            # already in assemble, no need to get from queue
            if queueName in self.assemble:
                continue

            while True:                  

                if m_queue.empty():
                    break

                imageMsg = m_queue.get()

                if self.enforce_sync and self.assemble_index < imageMsg.header.stamp:
                    # we shall throw away the assemble and start again
                    if self.assemble_index != -1:
                        print(f"reset index to {imageMsg.header.stamp}")

                    self.assemble_index = imageMsg.header.stamp
                    self.assemble = {}
                    self.assemble[queueName] = imageMsg
                    
                    continue
                elif self.enforce_sync and self.assemble_index > imageMsg.header.stamp:
                    # print(f"ignore {queueName} for later")
                    continue
                else:
                    self.assemble[queueName] = imageMsg
                    if self.enforce_sync:
                        break        
        
        # check for full assembly
        if len(self.assemble) == self.size:
            self.latest = self.assemble

            for cb in self.callbacks:
                cb(self.latest, self.assemble_index)

            if self.rolling:
                if self.synced_queue.full():
                    print(f"queue full: {self.queues.keys()}")
                    self.synced_queue.get()
                    self.synced_queue.put(self.assemble)
                else:
                    self.synced_queue.put(self.assemble, block=False)
            self.assemble = {}
            self.assemble_index = -1

    def callback(self, topic_type, topic_name, msg, ts):
        if self.queues[topic_name].full():
            self.queues[topic_name].get()
        
        self.queues[topic_name].put(msg)

        with self.lock:
            self.queue_update()

    # sub must have a `register_callback()` method
    def add_external_sub(self, sub, topic_name):
        self.queues[topic_name] = queue.Queue(10)
        self.size += 1
        sub.register_callback(self.callback)

    def register_callback(self, cb):
        self.callbacks.append(cb)

    def pop_latest(self):
        with self.lock:
            if self.latest == None:
                return {}
            else:
                return self.latest

    def pop_sync_queue(self):
        # not protected for read
        return self.synced_queue.get()

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def image_msg_to_cv_mat(imageMsg):
    if (imageMsg.encoding == "mono8"):

        mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
        mat = mat.reshape((imageMsg.height, imageMsg.width, 1))

        mat_vis = cv.cvtColor(mat, cv.COLOR_GRAY2BGR)

        return mat, mat_vis
    elif (imageMsg.encoding == "yuv420"):
        mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
        mat = mat.reshape((imageMsg.height * 3 // 2, imageMsg.width, 1))

        mat = cv.cvtColor(mat, cv.COLOR_YUV2BGR_IYUV)

        return mat, mat
    elif (imageMsg.encoding == "bgr8"):
        mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
        mat = mat.reshape((imageMsg.height, imageMsg.width, 3))
        return mat, mat
    elif (imageMsg.encoding == "jpeg"):
        mat_jpeg = np.frombuffer(imageMsg.data, dtype=np.uint8)
        mat = cv.imdecode(mat_jpeg, cv.IMREAD_COLOR)
        return mat, mat
    else:
        raise RuntimeError("unknown encoding: " + imageMsg.encoding)
    
def disparity_to_cv_mat(imageMsg):
    if (imageMsg.encoding == "disparity8"):

        mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
        mat = mat.reshape((imageMsg.height, imageMsg.width, 1))

        disp_vis = (mat * (255.0 / imageMsg.maxDisparity)).astype(np.uint8)
        disp_vis = cv.applyColorMap(disp_vis, cv.COLORMAP_JET)

        return mat, disp_vis
    
    elif (imageMsg.encoding == "disparity16"):
        mat_uint16 = np.frombuffer(imageMsg.data, dtype=np.uint16)
        mat_uint16 = mat_uint16.reshape((imageMsg.height, imageMsg.width, 1))

        mat_float32 = mat_uint16.astype(np.float32) / 8.0

        disp_vis = (mat_float32 * (255.0 / imageMsg.maxDisparity)).astype(np.uint8)
        disp_vis = cv.applyColorMap(disp_vis, cv.COLORMAP_JET)

        return mat_float32, disp_vis
    else:
        raise RuntimeError(f"disparity type not supported: {imageMsg.encoding}")


class VioSubscriber:

    def __init__(self, vio_topic):
        print(f"subscribing to viostate topic {vio_topic}")
        sub = CapnpSubscriber("Vio", vio_topic)
        sub.set_callback(self.callback)

        self.rolling = False
        self.m_queue = queue.Queue(2000)
        self.latest = None
        self.topic = vio_topic


    def register_callback(self, cb):
        self.vio_callbacks.append(cb)

    def callback(self, topic_type, topic_name, msg, ts):
        # need to remove the .decode() function within the Python API of ecal.core.subscriber ByteSubscriber

        # print(topic_type, topic_name, msg, ts, sep="\n")
        with eCALOdometry3d.Odometry3d.from_bytes(msg) as odometryMsg:

            # for cb in self.vio_callbacks:
            #     cb("capnp:Odometry3D", topic_name, odometryMsg, ts)

            self.latest = odometryMsg

            if self.rolling:
                try:
                    self.m_queue.put(odometryMsg, block=False)
                except queue.Full:
                    print("vio queue full")
                    # print(self.m_queue.qsize())
                    self.m_queue.get()
                    self.m_queue.put(odometryMsg)

    def pop_latest(self):
        # with self.lock:
        if self.latest == None:
            return {}
        else:
            return self.latest

    def pop_queue(self):
        # imu is quite frequent, it is ok to be blocked

        # if self.m_queue.qsize() > 10:
        #     print(self.m_queue.qsize())
        return self.m_queue.get()

            # # read in data
            # self.position_x = odometryMsg.pose.position.x
            # self.position_y = odometryMsg.pose.position.y
            # self.position_z = odometryMsg.pose.position.z

            # self.orientation_x = odometryMsg.pose.orientation.x
            # self.orientation_y = odometryMsg.pose.orientation.y
            # self.orientation_z = odometryMsg.pose.orientation.z
            # self.orientation_w = odometryMsg.pose.orientation.w

            # self.ts = odometryMsg.header.stamp
            # # text
            # self.header = odometryMsg.header
            # position_msg = f"position: \n {odometryMsg.pose.position.x:.4f}, {odometryMsg.pose.position.y:.4f}, {odometryMsg.pose.position.z:.4f}"
            # orientation_msg = f"orientation: \n  {odometryMsg.pose.orientation.w:.4f}, {odometryMsg.pose.orientation.x:.4f}, {odometryMsg.pose.orientation.y:.4f}, {odometryMsg.pose.orientation.z:.4f}"
            
            # device_latency_msg = f"device latency = {odometryMsg.header.latencyDevice / 1e6 : .2f} ms"
            
            # vio_host_latency = time.monotonic() *1e9 - odometryMsg.header.stamp 
            # host_latency_msg = f"host latency = {vio_host_latency / 1e6 :.2f} ms"
            
            # self.vio_msg = position_msg + "\n" + orientation_msg + "\n" + device_latency_msg + "\n" + host_latency_msg



class VioStateSubscriber:

    def __init__(self, viostate_topic):
        print(f"subscribing to viostate topic {viostate_topic}")
        sub = CapnpSubscriber("VioState", viostate_topic)
        sub.set_callback(self.callback)

        self.rolling = False
        self.m_queue = queue.Queue(2000)
        self.latest = None
        self.topic = viostate_topic


    def register_callback(self, cb):
        self.viviostate_callbacks.append(cb)

    def callback(self, topic_type, topic_name, msg, ts):
        # need to remove the .decode() function within the Python API of ecal.core.subscriber ByteSubscriber

        # print(topic_type, topic_name, ts, sep="\n")
        with eCALOdometry3d.VioState.from_bytes(msg) as viostateMsg:
            # print(viostateMsg.biasAccel, viostateMsg.biasGyro)
    #         print(viostateMsg.states)
    # #         # for cb in self.vio_callbacks:
    # #         #     cb("capnp:Odometry3D", topic_name, odometryMsg, ts)

            self.latest = viostateMsg

            if self.rolling:
                try:
                    self.m_queue.put(viostateMsg, block=False)
                except queue.Full:
                    print("vio queue full")
                    # print(self.m_queue.qsize())
                    self.m_queue.get()
                    self.m_queue.put(viostateMsg)

    def pop_latest(self):
        # with self.lock:
        if self.latest == None:
            return {}
        else:
            return self.latest

    def pop_queue(self):
        # imu is quite frequent, it is ok to be blocked

        # if self.m_queue.qsize() > 10:
        #     print(self.m_queue.qsize())
        return self.m_queue.get()

            # # read in data
            # self.position_x = odometryMsg.pose.position.x
            # self.position_y = odometryMsg.pose.position.y
            # self.position_z = odometryMsg.pose.position.z

            # self.orientation_x = odometryMsg.pose.orientation.x
            # self.orientation_y = odometryMsg.pose.orientation.y
            # self.orientation_z = odometryMsg.pose.orientation.z
            # self.orientation_w = odometryMsg.pose.orientation.w

            # self.ts = odometryMsg.header.stamp
            # # text
            # self.header = odometryMsg.header
            # position_msg = f"position: \n {odometryMsg.pose.position.x:.4f}, {odometryMsg.pose.position.y:.4f}, {odometryMsg.pose.position.z:.4f}"
            # orientation_msg = f"orientation: \n  {odometryMsg.pose.orientation.w:.4f}, {odometryMsg.pose.orientation.x:.4f}, {odometryMsg.pose.orientation.y:.4f}, {odometryMsg.pose.orientation.z:.4f}"
            
            # device_latency_msg = f"device latency = {odometryMsg.header.latencyDevice / 1e6 : .2f} ms"
            
            # vio_host_latency = time.monotonic() *1e9 - odometryMsg.header.stamp 
            # host_latency_msg = f"host latency = {vio_host_latency / 1e6 :.2f} ms"
            
            # self.vio_msg = position_msg + "\n" + orientation_msg + "\n" + device_latency_msg + "\n" + host_latency_msg






# Modified from https://github.com/matsuren/dscamera
class FisheyeDoubleSphere(object):
    """DSCamera class.
    V. Usenko, N. Demmel, and D. Cremers, "The Double Sphere Camera Model",
    Proc. of the Int. Conference on 3D Vision (3DV), 2018.
    """

    def __init__(self, width, height, fx, fy, cx, cy, xi, alpha, fov):
        # Fisheye camera parameters
        self.h, self.w = height, width
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.xi = xi
        self.alpha = alpha
        self.fov = float(fov)
        fov_rad = self.fov / 180 * np.pi
        self.fov_cos = np.cos(fov_rad / 2)
        self.intrinsic_keys = ["fx", "fy", "cx", "cy", "xi", "alpha"]

        # Valid mask for fisheye image
        self._valid_mask = None

    @property
    def img_size(self) -> Tuple[int, int]:
        return self.h, self.w

    @img_size.setter
    def img_size(self, img_size: Tuple[int, int]):
        self.h, self.w = map(int, img_size)

    @property
    def intrinsic(self) -> Dict[str, float]:
        intrinsic = {key: self.__dict__[key] for key in self.intrinsic_keys}
        return intrinsic

    @intrinsic.setter
    def intrinsic(self, intrinsic: Dict[str, float]):
        for key in self.intrinsic_keys:
            self.__dict__[key] = intrinsic[key]

    @property
    def valid_mask(self):
        if self._valid_mask is None:
            # Calculate and cache valid mask
            x = np.arange(self.w)
            y = np.arange(self.h)
            x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
            _, valid_mask = self.cam2world([x_grid, y_grid])
            self._valid_mask = valid_mask

        return self._valid_mask

    # def __repr__(self):
    #     return (
    #         f"[{self.__class__.__name__}]\n img_size:{self.img_size},fov:{self.fov},\n"
    #         f" intrinsic:{json.dumps(self.intrinsic, indent=2)}"
    #     )

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def cam2world(self, point2D):
        """cam2world(point2D) projects a 2D point onto the unit sphere.
        point3D coord: x:right direction, y:down direction, z:front direction
        point2D coord: x:row direction, y:col direction (OpenCV image coordinate)
        Parameters
        ----------
        point2D : numpy array or list([u,v])
            array of point in image
        Returns
        -------
        unproj_pts : numpy array
            array of point on unit sphere
        valid_mask : numpy array
            array of valid mask
        """
        # Case: point2D = list([u, v]) or np.array()
        if isinstance(point2D, (list, np.ndarray)):
            u, v = point2D
        # Case: point2D = list([Scalar, Scalar])
        if not hasattr(u, "__len__"):
            u, v = np.array([u]), np.array([v])

        # Decide numpy or torch
        if isinstance(u, np.ndarray):
            xp = np
        # else:
        #     xp = torch

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r2 = mx * mx + my * my

        # Check valid area
        s = 1 - (2 * self.alpha - 1) * r2
        valid_mask = s >= 0
        s[~valid_mask] = 0.0
        mz = (1 - self.alpha * self.alpha * r2) / (
            self.alpha * xp.sqrt(s) + 1 - self.alpha
        )

        mz2 = mz * mz
        k1 = mz * self.xi + xp.sqrt(mz2 + (1 - self.xi * self.xi) * r2)
        k2 = mz2 + r2
        k = k1 / k2

        # Unprojected unit vectors
        if xp == np:
            unproj_pts = k[..., np.newaxis] * np.stack([mx, my, mz], axis=-1)
        # else:
        #     unproj_pts = k.unsqueeze(-1) * torch.stack([mx, my, mz], dim=-1)
        unproj_pts[..., 2] -= self.xi

        # Calculate fov
        unprojected_fov_cos = unproj_pts[..., 2]  # unproj_pts @ z_axis
        fov_mask = unprojected_fov_cos >= self.fov_cos
        valid_mask *= fov_mask
        return unproj_pts, valid_mask

    def world2cam(self, point3D):
        """world2cam(point3D) projects a 3D point on to the image.
        point3D coord: x:right direction, y:down direction, z:front direction
        point2D coord: x:row direction, y:col direction (OpenCV image coordinate).
        Parameters
        ----------
        point3D : numpy array or list([x, y, z])
            array of points in camera coordinate
        Returns
        -------
        proj_pts : numpy array
            array of points in image
        valid_mask : numpy array
            array of valid mask
        """
        x, y, z = point3D[..., 0], point3D[..., 1], point3D[..., 2]
        # Decide numpy or torch
        if isinstance(x, np.ndarray):
            xp = np
        # else:
        #     xp = torch

        # Calculate fov
        point3D_fov_cos = point3D[..., 2]  # point3D @ z_axis
        fov_mask = point3D_fov_cos >= self.fov_cos

        # Calculate projection
        x2 = x * x
        y2 = y * y
        z2 = z * z
        d1 = xp.sqrt(x2 + y2 + z2)
        zxi = self.xi * d1 + z
        d2 = xp.sqrt(x2 + y2 + zxi * zxi)

        div = self.alpha * d2 + (1 - self.alpha) * zxi
        u = self.fx * x / div + self.cx
        v = self.fy * y / div + self.cy

        # Projected points on image plane
        if xp == np:
            proj_pts = np.stack([u, v], axis=-1)
        # else:
        #     proj_pts = torch.stack([u, v], dim=-1)

        # Check valid area
        if self.alpha <= 0.5:
            w1 = self.alpha / (1 - self.alpha)
        else:
            w1 = (1 - self.alpha) / self.alpha
        w2 = w1 + self.xi / xp.sqrt(2 * w1 * self.xi + self.xi * self.xi + 1)
        valid_mask = z > -w2 * d1
        valid_mask *= fov_mask

        return proj_pts, valid_mask

    def _warp_img(self, img, img_pts, valid_mask):
        # Remap
        img_pts = img_pts.astype(np.float32)
        out = cv.remap(
            img, img_pts[..., 0], img_pts[..., 1], cv.INTER_LINEAR
        )
        out[~valid_mask] = 0.0
        return out

    # def to_perspective(self, img, img_size=(600, 960), f=0.25):
    def to_perspective(self, img, img_size=(400, 640), f=0.25):
        # Generate 3D points
        h, w = img_size
        z = f * min(img_size)
        x = np.arange(w) - w / 2
        y = np.arange(h) - h / 2
        x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
        point3D = np.stack([x_grid, y_grid, np.full_like(x_grid, z)], axis=-1)

        # Project on image plane
        img_pts, valid_mask = self.world2cam(point3D)
        out = self._warp_img(img, img_pts, valid_mask)

        # start_y = 100  # Top-left y coordinate
        # start_x = 160   # Top-left x coordinate
        # width = 640     # Width of the cropped region
        # height = 400    # Height of the cropped region

        # out = out[start_y:start_y+height, start_x:start_x+width]
        return out

    def to_equirect(self, img, img_size=(256, 512)):
        # Generate 3D points
        h, w = img_size
        phi = -np.pi + (np.arange(w) + 0.5) * 2 * np.pi / w
        theta = -np.pi / 2 + (np.arange(h) + 0.5) * np.pi / h
        phi_xy, theta_xy = np.meshgrid(phi, theta, indexing="xy")

        x = np.sin(phi_xy) * np.cos(theta_xy)
        y = np.sin(theta_xy)
        z = np.cos(phi_xy) * np.cos(theta_xy)
        point3D = np.stack([x, y, z], axis=-1)

        # Project on image plane
        img_pts, valid_mask = self.world2cam(point3D)
        out = self._warp_img(img, img_pts, valid_mask)
        return out


class FisheyeKB4(object):
    """DSCamera class.
    V. Usenko, N. Demmel, and D. Cremers, "The Double Sphere Camera Model",
    Proc. of the Int. Conference on 3D Vision (3DV), 2018.
    """

    def __init__(self, width, height, fx, fy, cx, cy, k1, k2, k3, k4):
        # Fisheye camera parameters
        self.h = height
        self.w = width
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.instrincs_mat = np.array([[fx, 0.0, cx], 
                                        [0.0, fy, cy], 
                                        [0.0, 0.0, 1.0]])
        
        self.distortion_coeff = np.array([[k1], [k2], [k3], [k4]])

    # Define the projection function for Kannala-Brandt model
    def to_perspective(self, img):

        map1, map2 = cv.fisheye.initUndistortRectifyMap(self.instrincs_mat, 
                                                        self.distortion_coeff, 
                                                        np.eye(3), 
                                                        self.instrincs_mat, 
                                                        (self.w, self.h), 
                                                        cv.CV_16SC2)

        perspective_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LANCZOS4, borderMode=cv.BORDER_CONSTANT)
        perspective_img = cv.resize(perspective_img, (640, 400), interpolation = cv.INTER_LANCZOS4)

        return perspective_img
