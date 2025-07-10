#!/usr/bin/env python3

import sys
import threading

sys.path.append("/home/youngho/FoundationStereo/")
sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")
sys.path.append("~/perception/devel/lib/python")
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge

import torch
import trimesh

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import message_filters

from core.utils.utils import InputPadder
from core.foundation_stereo import *
from core.foundation_stereo import FoundationStereo
from Utils import *

import argparse
import os
from omegaconf import OmegaConf
import pdb



class RGBDSubscriber:
    def __init__(self, args=None):
        rospy.init_node('perception_fs_node', anonymous=True)

        self.left_image = None
        self.right_image = None
        self.left_image_ts = None
        self.right_image_ts = None
        self.sam_initialized = False
        self.fp_initialized = False
        self.bridge  = CvBridge()
        self.RUN_FS = False
        self.args = args
        self.setup_fs()
        self.padder = None

        # Foundation Pose

        self.intrinsic = None  # 3x3 np.array

        self.lockl = threading.Lock()
        self.lockr = threading.Lock()
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image, self.IR1_callback, queue_size = 1)
        rospy.Subscriber("/camera/infra2/image_rect_raw", Image, self.IR2_callback, queue_size = 1)
        #self.ts = message

        self.pcd_pub = rospy.Publisher('/perception/pointcloud', PointCloud2, queue_size = 1)


        cv2.namedWindow("disp")
        #cv2.setMouseCallback("RGB & Depth", self.mouse_callback)

    def camera_info_callback(self, msg):
        K = np.array(msg.K).reshape(3, 3)
        self.intrinsic = K

    def IR1_callback(self, msg):
        if msg.encoding != "mono8":
            rospy.logwarn(f"Unexpected encoding: {msg.encoding}")
            return
        if not self.lockl.acquire(blocking=False):
            #rospy.logwarn("inference is still processing")
            return

        h, w = msg.height, msg.width
        self.left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.left_image = cv2.resize(self.left_image, fx=self.args.scale, fy = self.args.scale, dsize= None)
        self.left_image_ts = torch.as_tensor(self.left_image).cuda().float()[None, None].repeat(1,3,1,1)
        #rospy.loginfo(f"left image got {self.left_image.shape}")
        cv2.imwrite("ir1_img.png", self.left_image)


    def IR2_callback(self, msg):
        if msg.encoding != "mono8":
            rospy.logwarn(f"Unexpected encoding: {msg.encoding}")
            return
        if not self.lockr.acquire(blocking=False):
            #rospy.logwarn("inference is still processing")
            return

        h, w = msg.height, msg.width
        self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.right_image = cv2.resize(self.right_image, fx = self.args.scale, fy = self.args.scale, dsize = None)
        self.right_image_ts = torch.as_tensor(self.right_image).cuda().float()[None,None].repeat(1,3,1,1)
        #rospy.loginfo(f"right image got {self.right_image.shape}")
        cv2.imwrite("ir2_img.png", self.right_image)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)

            rospy.loginfo(f"Clicked at ({x}, {y})")

            if self.rgb_image is not None and self.depth_image is not None:
                if x < self.rgb_image.shape[1] and y < self.rgb_image.shape[0]:
                    rgb = self.rgb_image[y, x]
                    depth_m = self.depth_image[y, x] / 1000.0
                    rospy.loginfo(f"RGB: {rgb}, Depth: {depth_m:.3f} m")

            self.sam_initialized = False
            self.fp_initialized = False
            self.RUN_FP =True

    def setup_fs(self):
        self.image_predictor = FoundationStereo(args)
        ckpt = torch.load(self.args['ckpt_dir'])
        logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
        self.image_predictor.load_state_dict(ckpt['model'])
        self.image_predictor.cuda().eval()

    def convert_numpy_to_pointcloud2(self, points, frame_id="map"):
        """
        points: numpy array of shape (N, 3) or (N, 6) if RGB included
        returns: sensor_msgs/PointCloud2 message
        """
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        if points.shape[1] == 6:
            fields += [
                PointField('r', 12, PointField.FLOAT32, 1),
                PointField('g', 16, PointField.FLOAT32, 1),
                PointField('b', 20, PointField.FLOAT32, 1),
            ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        return pc2.create_cloud(header, fields, points)

    def run(self):
        rate = rospy.Rate(5)

        mask = None
        self.RUN_FS = True
        while not rospy.is_shutdown():
            if self.left_image_ts is not None and self.right_image_ts is not None:
                rospy.loginfo("both image_detected")
                if self.padder is None:
                    self.padder = InputPadder(self.left_image_ts.shape, divis_by=32, force_square=False)

                H, W = self.left_image_ts.shape[-2:]
                li = self.left_image_ts
                ri = self.right_image_ts
                li, ri = self.padder.pad(li, ri)
                imgli_ori = self.left_image.copy()
                self.lockl.release()
                self.lockr.release()
                if self.RUN_FS:
                    with torch.inference_mode():
                        self.disp = self.image_predictor(li, ri, iters=self.args.valid_iters, test_mode=True)

                    rospy.loginfo("disparity got")
                    self.disp = self.padder.unpad(self.disp.float())
                    self.disp = self.disp.data.cpu().numpy().reshape(H, W)
                    disp_vis = (self.disp - self.disp.min()) / (
                                self.disp.max() - self.disp.min() + 1e-6)  # normalize to [0, 1]
                    disp_vis = (disp_vis * 255).astype(np.uint8)
                    cv2.imshow("disp", disp_vis)
                    cv2.waitKey(1)

                    if self.args.get_pc:
                        with open(args.intrinsic_file, 'r') as f:
                            lines = f.readlines()
                            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,
                                                                                                                 3)
                            baseline = float(lines[1])
                        K[:2] *= args.scale
                        depth = K[0,0]*baseline/self.disp
                        xyz_map = depth2xyzmap(depth, K)
                        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), imgli_ori.reshape(-1, 3))
                        keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (
                                    np.asarray(pcd.points)[:, 2] <= args.z_far)
                        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
                        pcd = pcd.select_by_index(keep_ids)
                        points = np.asarray(pcd.points)
                        colors = li[0].flatten(1,2).cpu().numpy().transpose(1,0)
                        colors = colors[keep_mask]
                        pc_combined = np.hstack((points, colors)).astype(np.float32)
                        msg = self.convert_numpy_to_pointcloud2(pc_combined, frame_id="camera_link")
                        rospy.loginfo("message sent")
                        self.pcd_pub.publish(msg)

            rate.sleep()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    code_dir = "/home/youngho/FoundationStereo"
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default=f'{code_dir}/assets/left.png', type=str)
    parser.add_argument('--right_file', default=f'{code_dir}/assets/right.png', type=str)
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/assets/K.txt', type=str,
                        help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/pretrained_models/11-33-40/model_best_bp2.pth', type=str,
                        help='pretrained model path')
    parser.add_argument('--out_dir', default=f'{code_dir}/output/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int,
                        help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int,
                        help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30,
                        help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    args = parser.parse_args()
    cfg = OmegaConf.load(f'{os.path.dirname(args.ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    rospy.loginfo(f"args: {args}")
    RGBDSubscriber(args).run()
