import sys

import cv2
from imread_from_url import imread_from_url

from detector import utils
from detector.YOLOv8ObjectDetector import YOLOv8ObjectDetector

# Initialize yolov8 object detector
model_path = "models/yolov8n.onnx"
yolov8_detector = YOLOv8ObjectDetector(model_path, conf_threshold=0.2, iou_threshold=0.3)

# Read image
img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
img = imread_from_url(img_url)

# Detect Objects
boxes, scores, class_ids = yolov8_detector.detect_objects(img)

# TODO: This might sum up the number of elements with label xy, which can then be passed to the other services
combined_img = utils.merge_image_with_overlay(img, boxes, scores, class_ids)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
sys.exit()
