import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_one_face(im):
    face_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)

    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]



if __name__ == '__main__':

   
    #tracking code

    cap = cv2.VideoCapture('vid5.mp4')
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('outputkalman&meanshift.avi',fourcc,-1, 20.0, (1280,720))

    frameCounter = 0
    # read first frame
    ret ,frame = cap.read()
    


    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    pt = (0,c+w/2,r+h/2)
    # Write track point for first frame

    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)
    state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    kalman = cv2.KalmanFilter(4,2,0)	
    kalman.transitionMatrix = np.array([[1., 0., .5, 0.],
                                    [0., 1., 0., .5],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    measurement = np.array([c+w/2, r+h/2], dtype='float64')
   
    
    while(1):
        ret ,frame = cap.read() # read another frame
        frame_expanded = np.expand_dims(frame, axis=0)
        if ret == False:
            break
           

        
    #kalman prediction
        prediction = kalman.predict() #prediction
        x,y,w,h = detect_one_face(frame) #checking measurement
        measurement = np.array([x+w/2, y+h/2], dtype='float64')
            
        if not (x ==0 and y==0 and w==0 and h ==0):
            posterior = kalman.correct(measurement)
        if x ==0 and y==0 and w==0 and h ==0:
            x,y,w,h = prediction
        else:
            x,y,w,h = posterior	
        pt = (frameCounter,x+w/2,y+h/2)
        print (pt)
        
        img2 = cv2.rectangle(frame, (int(x),int(y)), (int(x+20),int(y+20)), 255,2)
    
        cv2.imshow('img2',img2)
        out.write(frame)
        
        k = cv2.waitKey(25) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

		
        frameCounter = frameCounter + 1

    cv2.destroyAllWindows()
    out.release()
    cap.release()



