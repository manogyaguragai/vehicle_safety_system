import cv2
from ultralytics import YOLO
import math 

model = YOLO("/home/manogyaguragai/Downloads/bestmain.pt")

class_names= ['bicyclist', 'driver', 'helmet', 'no-helmet', 'notsure']

def detect_vehicle():
    # Initialize webcam capture
    webcam_capture = cv2.VideoCapture(0)  # capture frames from the default camera
    webcam_capture.set(3, 640)  # Set width of the window
    webcam_capture.set(4, 480)  # Set height of the window

    while True:
        success, img = webcam_capture.read()  # Read the frame from webcam
        if not success:
            break

        # results = model(img, stream=True)  # Get predictions from the model
        # Convert BGR (OpenCV format) to RGB (format expected by most models)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, stream=True)

        # Parse results
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Draw bounding box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence score
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence: ", confidence)

                # Class name based on the prediction (0: With Helmet, 1: Without Helmet)
                cls = int(box.cls[0])  # Get the class index
                class_name = class_names[cls]  # Get the class name from the list
                print("Class name: ", class_name)

                # Display the class name on the image
                org = (x1, y1 - 10)  # Position to place text slightly above the bounding box
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)  # Blue text color
                thickness = 2

                cv2.putText(img, f'{class_name} ({confidence:.2f})', org, font, fontScale, color, thickness)

        # Display the webcam feed with detections
        cv2.imshow('Webcam', img)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    webcam_capture.release()
    cv2.destroyAllWindows()