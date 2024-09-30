import cv2

def capture_from_webcam():
  webcam_capture = cv2.VideoCapture(0)  # capture frames from the default camera
  webcam_capture.set(3, 640)
  webcam_capture.set(4, 480)

  while True:
    ret, frame = webcam_capture.read()
    cv2.imshow('Webcam Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):   # 0xFF is 255 in binary, and a AND operation is performed to return only lower 8 bits of the result
      break   # exit when user presses 'q' key
    
  webcam_capture.release()
  cv2.destroyAllWindows()