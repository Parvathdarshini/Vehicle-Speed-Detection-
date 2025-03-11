#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dlib
import cv2
import time
from datetime import datetime
import os
import numpy as np

carCascade = cv2.CascadeClassifier('HaarCascadeClassifier.xml')

# TAKE VIDEO 
video = cv2.VideoCapture('videoTest.mp4')

WIDTH = 1280  # WIDTH OF VIDEO FRAME
HEIGHT = 720  # HEIGHT OF VIDEO FRAME
cropBegin = 240  # CROP VIDEO FRAME FROM THIS POINT
mark1 = 120  # MARK TO START TIMER
mark2 = 360  # MARK TO END TIMER
markGap = 15  # DISTANCE IN METRES BETWEEN THE MARKERS
fpsFactor = 3  # TO COMPENSATE FOR SLOW PROCESSING
speedLimit = 20  # SPEED LIMIT
startTracker = {}  # STORE STARTING TIME OF CARS
endTracker = {}  # STORE ENDING TIME OF CARS

# MAKE DIRECTORY TO STORE OVER-SPEEDING CAR IMAGES
if not os.path.exists('overspeeding/cars/'):
    os.makedirs('overspeeding/cars/')

print('Speed Limit Set at 20 Kmph')

def blackout(image):
    xBlack = 360
    yBlack = 300
    triangle_cnt = np.array([[0, 0], [xBlack, 0], [0, yBlack]])
    triangle_cnt2 = np.array([[WIDTH, 0], [WIDTH - xBlack, 0], [WIDTH, yBlack]])
    cv2.drawContours(image, [triangle_cnt], 0, (0, 0, 0), -1)
    cv2.drawContours(image, [triangle_cnt2], 0, (0, 0, 0), -1)
    return image

# FUNCTION TO SAVE CAR IMAGE, DATE, TIME, SPEED
def saveCar(speed, image):
    now = datetime.today().now()
    nameCurTime = now.strftime("%d-%m-%Y-%H-%M-%S-%f")
    # Save as PNG for better quality
    link = 'overspeeding/cars/' + nameCurTime + '.png'
    
    # Resize the image to a larger size if necessary
    high_res_image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    
    # Save the image with higher quality
    cv2.imwrite(link, high_res_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # For PNG, 0 is no compression, which is the best quality

# FUNCTION TO CALCULATE SPEED
def estimateSpeed(carID):
    timeDiff = endTracker[carID] - startTracker[carID]
    speed = round(markGap / timeDiff * fpsFactor * 3.6, 2)
    return speed

# FUNCTION TO TRACK CARS
def trackMultipleObjects():
    frameCounter = 0
    currentCarID = 0
    carTracker = {}

    while True:
        rc, image = video.read()
        if not rc:
            print("Failed to read the frame. Exiting...")
            break

        frameTime = time.time()
        image = cv2.resize(image, (WIDTH, HEIGHT))[cropBegin:720, 0:1280]
        resultImage = blackout(image)
        cv2.line(resultImage, (0, mark1), (1280, mark1), (0, 0, 255), 2)
        cv2.line(resultImage, (0, mark2), (1280, mark2), (0, 0, 255), 2)

        frameCounter += 1

        # DELETE CAR IDs NOT IN FRAME
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)

        # MAIN PROGRAM
        if frameCounter % 60 == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))  # DETECT CARS IN FRAME

            for (_x, _y, _w, _h) in cars:
                # GET POSITION OF A CAR
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                xbar = x + 0.5 * w
                ybar = y + 0.5 * h

                matchCarID = None

                # IF CENTROID OF CURRENT CAR NEAR THE CENTROID OF ANOTHER CAR IN PREVIOUS FRAME THEN THEY ARE THE SAME
                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    tx = int(trackedPosition.left())
                    ty = int(trackedPosition.top())
                    tw = int(trackedPosition.width())
                    th = int(trackedPosition.height())

                    txbar = tx + 0.5 * tw
                    tybar = ty + 0.5 * th

                    if ((tx <= xbar <= (tx + tw)) and (ty <= ybar <= (ty + th)) and (x <= txbar <= (x + w)) and (y <= tybar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker

                    currentCarID += 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            tx = int(trackedPosition.left())
            ty = int(trackedPosition.top())
            tw = int(trackedPosition.width())
            th = int(trackedPosition.height())

            # ESTIMATE SPEED
            speed = None
            if carID not in startTracker and mark2 > ty + th > mark1 and ty < mark1:
                startTracker[carID] = frameTime

            elif carID in startTracker and carID not in endTracker and mark2 < ty + th:
                endTracker[carID] = frameTime
                speed = estimateSpeed(carID)
                if speed > speedLimit:
                    print(f'CAR-ID : {carID} : {speed} kmph - OVERSPEED')
                    saveCar(speed, image[ty:ty + th, tx:tx + tw])

            # PUT BOUNDING BOXES
            if speed is None:
                rectangleColor = (0, 255, 0)  # Green for cars still being tracked but not yet evaluated
            elif speed > speedLimit:
                rectangleColor = (0, 0, 255)  # Red for overspeeding cars
            else:
                rectangleColor = (0, 255, 0)  # Green for non-overspeeding cars

            cv2.rectangle(resultImage, (tx, ty), (tx + tw, ty + th), rectangleColor, 2)
            cv2.putText(resultImage, str(carID), (tx, ty - 5), cv2.FONT_HERSHEY_DUPLEX, 1, rectangleColor, 1)

        # DISPLAY EACH FRAME
        cv2.imshow('result', resultImage)

        # Check if window should close (ESC key press)
        if cv2.waitKey(1) == 27:
            print("Escape key pressed. Exiting...")
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    trackMultipleObjects()


# In[ ]:




