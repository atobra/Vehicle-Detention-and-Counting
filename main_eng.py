import cv2
import numpy as np
from time import sleep

min_width = 80 #Minimum rectangular width
min_height = 80 #Minimum rectangular height

offset = 6 #Error allowed between pixel

line_position = 550 #Count line position

delay = 60 #Video fps

detector = []
cars = 0

cap = cv2.VideoCapture('video.mp4')      #Import video file
subtraction = cv2.bgsegm.createBackgroundSubtractorMOG()    #Extracts vehicle images from rest of video

def all_lanes():
        def position(x_start, x_end, y_axis, frame):
            cv2.line(frame, (x_start, y_axis), (x_end, y_axis), (255,127,0), 3) 

        static_line = position(0, 1200, line_position, frame1)  #Changes the position of line beyond which counting is done

        #Loop for counting the number of cars that have passed
        for (x,y) in detector:
            if y<(line_position + offset) and y>(line_position - offset):   #Definition of counting region in the y-axis
                global cars
                cars += 1
                static_line  
                detector.remove((x,y))
                print("car is detected : "+str(cars))
                

def get_center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy


while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo)

    #Generation of gray video from original video
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtraction.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilated = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)
    contour,h = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    #Loop for vehicle detection and verification
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_contour = (w >= min_width) and (h >= min_height)
        if not validate_contour:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        center = get_center(x, y, w, h)
        detector.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1)

        all_lanes()
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)    #Display vehicle count in video
    cv2.imshow("Video Original" , frame1)   #Title of video count window
    cv2.imshow("Detector",dilated)      #Vehicle detection video

    frameTime = 10
    if cv2.waitKey(frameTime) & 0xFF == ord('q'):
        break

cap.release()    
cv2.destroyAllWindows()
