import cv2
import numpy as np
import copy
import serial
import time

mctrl = serial.Serial('COM3', 115200, timeout=.1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)

count = 0
index = 0

while True :
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #OpenCV uses BGR format. Not RGB
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #change values later according to the image you get
    cv2.line(img,(320,0),(320,480),(255,255,255),2)
    cv2.line(img,(220,0),(220,480),(0,0,255),2)
    cv2.line(img,(420,0),(420,480),(0,0,255),2)
    


    
    for (x,y,w,h) in faces:
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2) #rectangle(drawn on img, str point, end point, rectangle color in BGR, line width)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        

        centre = x+(w/2) #define center point using x value only (Horizontal movement)
        motor control signal write
        if centre<220 :
            mctrl.write('0')
        elif centre>420 :
            mctrl.write('2')
        else :
            mctrl.write('1')
            
        count+=1
        cv2.imshow('Face_Detection', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()

