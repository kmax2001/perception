#!/usr/bin/env python3

import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image
from sam2.build_sam import build_sam2_camera_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class RGBDSubscriber:
    def __init__(self):
        rospy.init_node('rgbd_manual_client_node', anonymous=True)

        self.rgb_image = None
        self.depth_image = None
        self.clicked_point = None
        self.sam_initialized = False

        self.setup_sam()

        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        cv2.namedWindow("RGB & Depth")
        cv2.setMouseCallback("RGB & Depth", self.mouse_callback)

    def rgb_callback(self, msg):
        if msg.encoding != "bgr8":
            rospy.logwarn(f"Unexpected encoding: {msg.encoding}")
            return

        h, w = msg.height, msg.width
        self.rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))

    def depth_callback(self, msg):
        if msg.encoding != "16UC1":
            rospy.logwarn(f"Unexpected encoding: {msg.encoding}")
            return

        h, w = msg.height, msg.width
        self.depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w))

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

    def setup_sam(self):
        SAM_CHECKPOINT = "./checkpoints/sam2_hiera_tiny.pt"
        SAM_MODEL_CONFIG = "sam2_hiera_t.yaml"
        self.predictor = build_sam2_camera_predictor(SAM_MODEL_CONFIG, SAM_CHECKPOINT)
        self.image_predictor = SAM2ImagePredictor(build_sam2(SAM_MODEL_CONFIG, SAM_CHECKPOINT, device="cuda"))

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.rgb_image is not None and self.depth_image is not None:
                rgb = self.rgb_image.copy()
                depth_vis = cv2.convertScaleAbs(self.depth_image, alpha=0.03)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                vis_img = np.concatenate((rgb, depth_colored), axis=1)

                if self.clicked_point:
                    x, y = self.clicked_point
                    cv2.circle(vis_img, (x, y), 5, (0, 0, 255), -1)

                    points = np.array([[x, y]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)

                    if not self.sam_initialized:
                        self.sam_initialized = True
                        self.predictor._init_state()
                        self.predictor.load_first_frame(rgb)
                        _, _, _ = self.predictor.add_new_points(frame_idx=0, obj_id=1, points=points, labels=labels)

                    self.clicked_point = None


                if self.sam_initialized:
                    out_obj_ids, out_mask_logits = self.predictor.track(rgb)
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                    mask_rgb = self.rgb_image.copy()
                    mask_rgb[mask==False] = 255
                    vis_img = np.concatenate((vis_img, mask_rgb), axis=1)

                cv2.imshow("RGB & Depth", vis_img)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

            rate.sleep()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RGBDSubscriber().run()
