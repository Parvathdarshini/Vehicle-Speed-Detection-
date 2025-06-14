# Vehicle Speed Detection System
This project aims to detect and track vehicles in a video stream, calculate their speed, and identify overspeeding vehicles. If a vehicle exceeds the specified speed limit, the system automatically captures and stores an image of that vehicle for records or further analysis.

## Features:
- Real-time vehicle detection using computer vision techniques
- Object tracking to monitor vehicles across frames
- Speed calculation based on frame difference and known distances
- Automatic image capture of overspeeding vehicles
- Simple and clean output interface

## Tech Stack:
- Language: Python
- Libraries: OpenCV, NumPy, imutils, datetime, math and CV2

## How It Works:
- Vehicle Detection: The system uses contour detection and background subtraction techniques via OpenCV to detect moving vehicles in a video.
- Tracking: Once detected, vehicles are tracked between two reference lines.
- Speed Calculation: Speed is estimated using the time taken to move between two points, and a known physical distance between those points.
- Overspeed Alert: If a vehicle exceeds the defined speed limit, its image is captured and saved automatically with the timestamp.

## How to Run:
- Clone the repository:
git clone https://github.com/Parvathdarshini/Vehicle-Speed-Detection-.git  and
cd Vehicle-Speed-Detection-

- Install the required packages:
pip install opencv-python imutils numpy

- Run the script:
python speed_detection.py

- Make sure you have the input_video.mp4 file in the same directory or update the script to point to your own video file.

## Configuration:
- You can configure the speed limit and reference line positions in the code: speed_limit = 60  # km/h.
- Also, update the distance (in meters) between reference lines based on your video footage.
- Sample Output: Detected overspeeding vehicle images are saved in the vehicle_images/ folder with timestamps.

## Future Improvements:
- Integrate deep learning-based vehicle detection (YOLO, SSD, etc.) for more accuracy
- Add license plate recognition for identifying vehicles
- Build a dashboard for live monitoring
- Export data to a CSV or database
