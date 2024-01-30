import time

import cv2

from detector import utils
from detector.DeviceMetricReporter import DeviceMetricReporter
from detector.ServiceMetricReporter import ServiceMetricReporter
from detector.YOLOv8ObjectDetector import YOLOv8ObjectDetector

# Benchmark for road race with 'video.mp4'
# PC GPU --> 64 FPS
# Laptop CPU --> 15 / 24 FPS
# Orin GPU --> 35 FPS
# Xavier GPU --> 34 FPS
# Xavier CPU --> 4 FPS

# cpu = Gauge('cpu', 'Description of gauge')
device_metric_reporter = DeviceMetricReporter(clear_collection=True)
provider_metric_reporter = ServiceMetricReporter("Provider", clear_collection=True)
# cap = cv2.VideoCapture("data/video.mp4")

# videoUrl = 'https://youtu.be/Snyg0RqpVxY'
# cap = cap_from_youtube(videoUrl, resolution='720p')
# start_time = 5 # skip first {start_time} seconds
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))
# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))

model_path = "models/yolov8n.onnx"
detector = YOLOv8ObjectDetector(model_path, conf_threshold=0.5, iou_threshold=0.5)
simulate_fps = True
cv2.namedWindow("Detected Objects", cv2.WINDOW_AUTOSIZE)


def process_video(video_path, video_info, show_result=False, repeat=1):
    for source_fps in video_info:
        for x in range(repeat):

            print(f"Now processing: {source_fps} Round {x + 1}")
            available_time_frame = (1000 / source_fps)
            cap = cv2.VideoCapture("data/original_cut.mp4")
            # cap = cv2.VideoCapture(video_path + source_res + "_" + str(source_fps) + ".mp4")
            if not cap.isOpened():
                print("Error opening video ...")
                return

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

                start_time = time.time()
                boxes, scores, class_ids = detector.detect_objects(frame)
                combined_img = utils.merge_image_with_overlay(frame, boxes, scores, class_ids)

                if show_result:
                    cv2.imshow("Detected Objects", combined_img)

                processing_time = (time.time() - start_time) * 1000.0
                pixel = combined_img.shape[0]

                # TODO: Report device and SLO at the same time
                blanket_a = device_metric_reporter.create_metrics()
                blanket_b = provider_metric_reporter.create_metrics(processing_time, source_fps, pixel)

                intersection_name = utils.sort_and_join(blanket_a["target"], blanket_b["target"])
                merged_metrics = utils.merge_single_dicts(blanket_a["metrics"], blanket_b["metrics"])
                device_metric_reporter.report_metrics(intersection_name, merged_metrics)

                if simulate_fps:
                    if processing_time < available_time_frame:
                        time.sleep((available_time_frame - processing_time) / 1000)

    detector.print_benchmark()


if __name__ == "__main__":
    process_video(video_path="../video_data/",
                  video_info=[4, 8, 12, 16],
                  show_result=True)
