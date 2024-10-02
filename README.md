# Vehicle Safety System

This project aims to detect vehicles using a YOLO (You Only Look Once) model and a webcam feed. The system identifies vehicles, checks whether seatbelts (for cars) or helmets (for motorbikes) are being used, and logs the vehicle's number plate in case of violations.

## Features:
- Detects various types of vehicles.
- Checks for seatbelt use (for cars) and helmet presence (for motorbikes).
- Violations (no seatbelt or helmet) are recorded along with the vehicle’s number plate.

## Prerequisites
- Python 3.x
- `opencv-python` (`cv2`)
- `ultralytics` (for YOLO model)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/manogyaguragai/vehicle_safety_system.git
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Make sure you have a working webcam or camera connected.
2. Run the script to start detecting vehicles:
   ```bash
   python3 main.py
   ```
3. The program will open a window displaying the camera feed with bounding boxes around detected objects. It will also print the detected vehicle’s class name (car, motorbike, etc.) and the confidence level.

4. Press `q` to exit the webcam window and stop the detection.

## How it works
- The program captures frames from your webcam and passes them through the YOLO model for object detection.
- Bounding boxes are drawn around detected objects, and the class name and confidence are printed.
- The program can be further developed to check for seatbelts and helmets and log violations.

## Model Used
- The project uses the YOLO model (`yolo11m.pt`) to detect objects in the webcam feed.


---
Feel free to reach out if any improvements can be done!


