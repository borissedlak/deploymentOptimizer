import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("data/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Fails for the image test here
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    if cv2.waitKey(1) == ord('q'):
        break

    success, frame = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(frame, persist=True, show=False)

    frame = counter.start_counting(frame, tracks)

cap.release()
cv2.destroyAllWindows()
