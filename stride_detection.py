import sys, cv2
sys.path.insert(0, '.')
from ultralytics import YOLO

# Force file backend — not GStreamer
cap = cv2.VideoCapture('stride_detection.mp4', cv2.CAP_FFMPEG)
model = YOLO('yolov8n-pose.pt')
frame_num = 0

while frame_num < 150:
    ret, frame = cap.read()
    if not ret: break
    frame_num += 1
    if frame_num % 3 != 0: continue

    # i dont want to rotate the video 90 degrees but I will resize the image down
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(frame, (1280, 720))

    results = model.predict(frame, verbose=False, conf=0.3)
    if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
        kps = results[0].keypoints[0]
        la, ra = kps[5], kps[6]  # left and right ankle
        print(f'f={frame_num:03d} LA_y={la[1]:.0f}(c={la[2]:.2f}) RA_y={ra[1]:.0f}(c={ra[2]:.2f}) hip_y={lh[1]:.0f} dy={abs(la[1]-lh[1]):.0f}')
    else:
        print(f'f={frame_num:03d} NO DETECTION')  # no keypoints detected
    cap.release()

quit()
print('Video opened:', cap.isOpened())
print('Total frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('FPS:', cap.get(cv2.CAP_PROP_FPS))
print('Resolution:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()