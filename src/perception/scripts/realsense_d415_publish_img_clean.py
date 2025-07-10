#!/usr/bin/env python3

import rospy
import pyrealsense2 as rs
import numpy as np
from sensor_msgs.msg import Image as ROSImage

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo

def create_camera_info_msg(intrinsics, frame_id):
    msg = CameraInfo()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height = intrinsics.height
    msg.width = intrinsics.width
    msg.distortion_model = "plumb_bob"
    msg.D = list(intrinsics.coeffs)  # [k1, k2, p1, p2, k3]
    msg.K = [intrinsics.fx, 0, intrinsics.ppx,
             0, intrinsics.fy, intrinsics.ppy,
             0, 0, 1]
    msg.R = [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
    msg.P = [intrinsics.fx, 0, intrinsics.ppx, 0,
             0, intrinsics.fy, intrinsics.ppy, 0,
             0, 0, 1, 0]
    return msg

def create_pointcloud_msg(points, colors):
    # Filter out invalid points
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    colors = colors[mask]

    # Combine XYZ + RGB into a single list of tuples
    rgb_int = (colors[:, 0].astype(np.uint32) << 16) | \
              (colors[:, 1].astype(np.uint32) << 8) | \
              (colors[:, 2].astype(np.uint32))
    cloud_data = [(*p, rgb) for p, rgb in zip(points, rgb_int)]

    fields = [
        PointField('x', 0,  PointField.FLOAT32, 1),
        PointField('y', 4,  PointField.FLOAT32, 1),
        PointField('z', 8,  PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]

    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "camera_link"  # or use your actual frame

    return point_cloud2.create_cloud(header, fields, cloud_data)

def create_image_msg(image, encoding):
    msg = ROSImage()
    msg.header.stamp = rospy.Time.now()
    msg.height, msg.width = image.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = image.shape[1] * image.itemsize * (3 if encoding == "bgr8" else 1)
    msg.data = image.tobytes()
    return msg

def main():
    rospy.init_node('realsense_d415_rgbd_ros')

    rgb_pub = rospy.Publisher('/camera/color/image_raw', ROSImage, queue_size=10)
    depth_pub = rospy.Publisher('/camera/depth/image_raw', ROSImage, queue_size=10)
    pc_pub = rospy.Publisher('/camera/depth/color/points', PointCloud2, queue_size=10)

    rgb_info_pub = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=10)
    depth_info_pub = rospy.Publisher('/camera/depth/camera_info', CameraInfo, queue_size=10)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()

        if not depth or not color:
            continue

        color_np = np.asanyarray(color.get_data())
        depth_np = np.asanyarray(depth.get_data())

        color_intr = color.get_profile().as_video_stream_profile().get_intrinsics()
        depth_intr = depth.get_profile().as_video_stream_profile().get_intrinsics()

        # Publish RGB and Depth
        rgb_pub.publish(create_image_msg(color_np, "bgr8"))
        depth_pub.publish(create_image_msg(depth_np, "16UC1"))

        rgb_info_pub.publish(create_camera_info_msg(color_intr, frame_id="camera_color_optical_frame"))
        depth_info_pub.publish(create_camera_info_msg(depth_intr, frame_id="camera_depth_optical_frame"))

        if False:
            # Compute point cloud
            pc.map_to(color)
            points = pc.calculate(depth)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

            # Get corresponding RGB values
            h, w, _ = color_np.shape
            u = (tex_coords[:, 0] * w).astype(np.int32)
            v = (tex_coords[:, 1] * h).astype(np.int32)
            u = np.clip(u, 0, w - 1)
            v = np.clip(v, 0, h - 1)
            rgb_vals = color_np[v, u]  # shape: (N, 3)

            pc_msg = create_pointcloud_msg(verts, rgb_vals)
            pc_pub.publish(pc_msg)

        rate.sleep()

    pipeline.stop()

if __name__ == "__main__":
    main()
