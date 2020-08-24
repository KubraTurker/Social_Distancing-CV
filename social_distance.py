### REFERENCES
### https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
### https://docs.opencv.org/4.3.0/d6/d0f/group__dnn.html
### https://www.ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/
### https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/


import cv2
import numpy as np
from math import pow, sqrt

#Constant Values
preprocessing = False
calculateConstant_x = 300
calculateConstant_y = 615
personLabelID = 15.00
debug = True
accuracyThreshold = 0.4
RED = (0,0,255)
YELLOW = (0,255,255)
GREEN = (0,255,0)
write_video = False

# I used CLAHE preprocessing algorithm for detect humans better.
# HSV (Hue, Saturation, and Value channel). CLAHE uses value channel.
# Value channel refers to the lightness or darkness of a colour. An image without hue or saturation is a grayscale image.
def CLAHE(bgr_image: np.array) -> np.array:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def centroid(startX,endX,startY,endY):
    centroid_x = round((startX+endX)/2,4)
    centroid_y = round((startY+endY)/2,4)
    bboxHeight = round(endY-startY,4)
    return centroid_x,centroid_y,bboxHeight

def calcDistance(bboxHeight):
    distance = (calculateConstant_x * calculateConstant_y) / bboxHeight
    return distance

def drawResult(frame,position):
    for i in position.keys():
        if i in highRisk:
            rectangleColor = RED
        elif i in mediumRisk:
            rectangleColor = YELLOW
        else:
            rectangleColor = GREEN
        (startX, startY, endX, endY) = detectionCoordinates[i]

        cv2.rectangle(frame, (startX, startY), (endX, endY), rectangleColor, 2)


if __name__== "__main__":

    caffeNetwork = cv2.dnn.readNetFromCaffe("./SSD_MobileNet_prototxt.txt", "./SSD_MobileNet.caffemodel")
    cap = cv2.VideoCapture("./pedestrians.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_movie = cv2.VideoWriter("./result.avi", fourcc, 24, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


    while cap.isOpened():

        debug_frame, frame = cap.read()
        highRisk = set()
        mediumRisk = set()
        position = dict()
        detectionCoordinates = dict()

        if not debug_frame:
            print("Video cannot opened or finished!")
            break

        if preprocessing:
            frame = CLAHE(frame)

        (imageHeight, imageWidth) = frame.shape[:2]
        pDetection = cv2.dnn.blobFromImage(cv2.resize(frame, (imageWidth, imageHeight)), 0.007843, (imageWidth, imageHeight), 127.5)

        caffeNetwork.setInput(pDetection)
        detections = caffeNetwork.forward()

        for i in range(detections.shape[2]):

            accuracy = detections[0, 0, i, 2]
            if accuracy > accuracyThreshold:
                # Detection class and detection box coordinates.
                idOfClasses = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                (startX, startY, endX, endY) = box.astype('int')

                if idOfClasses == personLabelID:
                    # Default drawing bounding box.
                    bboxDefaultColor = (255,255,255)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), bboxDefaultColor, 2)
                    detectionCoordinates[i] = (startX, startY, endX, endY)

                    # Centroid of bounding boxes
                    centroid_x, centroid_y, bboxHeight = centroid(startX,endX,startY,endY)                    
                    distance = calcDistance(bboxHeight)
                    # Centroid in centimeter distance
                    centroid_x_centimeters = (centroid_x * distance) / calculateConstant_y
                    centroid_y_centimeters = (centroid_y * distance) / calculateConstant_y
                    position[i] = (centroid_x_centimeters, centroid_y_centimeters, distance)

        #Risk Counter Using Distance of Positions
        for i in position.keys():
            for j in position.keys():
                if i < j:
                    distanceOfBboxes = sqrt(pow(position[i][0]-position[j][0],2) 
                                          + pow(position[i][1]-position[j][1],2) 
                                          + pow(position[i][2]-position[j][2],2)
                                          )
                    if distanceOfBboxes < 150: # 150cm or lower
                        highRisk.add(i),highRisk.add(j)
                    elif distanceOfBboxes < 200 > 150: # between 150 and 200
                        mediumRisk.add(i),mediumRisk.add(j) 
       

        cv2.putText(frame, "Person in High Risk : " + str(len(highRisk)) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Person in Medium Risk : " + str(len(mediumRisk)) , (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, "Detected Person : " + str(len(detectionCoordinates)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        drawResult(frame, position)
        if write_video:            
            output_movie.write(frame)
        cv2.imshow('Result', frame)
        waitkey = cv2.waitKey(1)
        if waitkey == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

