import cv2
import os
import time
from ultralytics import YOLO
import math
import pyttsx3

engine = pyttsx3.init()

model = YOLO("/home/manogyaguragai/Downloads/bestmain.pt")

class_names = ['bicyclist', 'driver', 'helmet', 'no-helmet', 'notsure']

save_folder = "no_helmet_captured"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

no_helmet_start_time = None
image_count = 0
image_limit = 30
capture_interval = 5 

def detect_vehicle():
    global no_helmet_start_time, image_count

    webcam_capture = cv2.VideoCapture(0)
    webcam_capture.set(3, 640) 
    webcam_capture.set(4, 480)  

    while True:
        success, img = webcam_capture.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, stream=True)

        no_helmet_detected = False  

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])  
                class_name = class_names[cls] 
                print("Class name: ", class_name, "Confidence: ", confidence)

                if class_name == "no-helmet":
                    no_helmet_detected = True

                org = (x1, y1 - 10)
                cv2.putText(img, f'{class_name} ({confidence:.2f})', org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if no_helmet_detected:
            if no_helmet_start_time is None:
                no_helmet_start_time = time.time()
            else:
                elapsed_time = time.time() - no_helmet_start_time
                if elapsed_time > capture_interval and image_count < image_limit:
                    image_path = os.path.join(save_folder, f"no_helmet_{image_count + 1}.jpg")
                    cv2.imwrite(image_path, img)
                    print(f"Captured and saved: {image_path}")
                    
                    engine.say("Helmet Violation detected")
                    engine.runAndWait()

                    image_count += 1
                    no_helmet_start_time = time.time()  
        else:
            no_helmet_start_time = None 

        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) == ord('q'):
            break

    webcam_capture.release()
    cv2.destroyAllWindows()
