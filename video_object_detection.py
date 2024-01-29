import time

import cv2

from yolov8 import utils
from yolov8.CustomMetricReporter import CustomMetricReporter
from yolov8.DeviceMetricReporter import DeviceMetricReporter
from yolov8.YOLOv8ObjectDetector import YOLOv8ObjectDetector

# Benchmark for road race with 'video.mp4'
# PC GPU --> 64 FPS
# Laptop CPU --> 15 / 24 FPS
# Orin GPU --> 35 FPS
# Xavier GPU --> 34 FPS
# Xavier CPU --> 4 FPS

# cpu = Gauge('cpu', 'Description of gauge')
device_metric_reporter = DeviceMetricReporter("Laptop", clear_collection=True)
provider_metric_reporter = CustomMetricReporter("Provider", clear_collection=True)
cap = cv2.VideoCapture("data/video.mp4")

# videoUrl = 'https://youtu.be/Snyg0RqpVxY'
# cap = cap_from_youtube(videoUrl, resolution='720p')
# start_time = 5 # skip first {start_time} seconds
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))
# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))

model_path = "models/yolov8n.onnx"
detector = YOLOv8ObjectDetector(model_path, conf_threshold=0.5, iou_threshold=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # TODO: What could be interesting here? I can extract all this exactly here
    # --> Processing time per frame, fps, resolution

    start_time = time.time()
    boxes, scores, class_ids = detector.detect_objects(frame)
    combined_img = utils.merge_image_with_overlay(frame, boxes, scores, class_ids)
    cv2.imshow("Detected Objects", combined_img)

    processing_time = (time.time() - start_time) * 1000.0
    fps = 15  # Must actually replay in correct FPS
    pixel = combined_img.shape[0]

    # TODO: Add some SLO-relevant metrics
    # TODO: Report device and SLO at the same time
    device_metric_reporter.report_now()
    provider_metric_reporter.report_this(processing_time, fps, pixel)

    # out.write(combined_img)

detector.print_benchmark()
# out.release()
