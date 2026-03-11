import cv2
pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=BGRx ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=1"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
ret, frame = cap.read()
print("Success:", ret, frame.shape if ret else "no frame")
