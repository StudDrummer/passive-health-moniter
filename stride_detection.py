import sys, cv2
sys.path.insert(0, '.')

# Force file backend — not GStreamer
cap = cv2.VideoCapture('stride_detection.mp4', cv2.CAP_FFMPEG)
print('Video opened:', cap.isOpened())
print('Total frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('FPS:', cap.get(cv2.CAP_PROP_FPS))
print('Resolution:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()