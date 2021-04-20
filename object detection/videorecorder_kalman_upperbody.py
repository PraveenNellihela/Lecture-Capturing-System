import os
import sys
import cv2
import numpy as np
import threading
import tensorflow as tf
import queue
import logging
import time
import imutils
import serial
import copy
import tkinter as tk
from time import gmtime, strftime
from datetime import datetime, timedelta
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyaudio as pya
import wave

from utils import label_map_util
from utils import visualization_utils as vis_util
from msvcrt import getch
from PIL import ImageGrab
from imutils.video import VideoStream



############################### Screen capture thread ##############################
def screen_capture():
    
    logging.debug('Starting')
    while run_code:
        print('running screen cap using if true')
        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        screenout = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)

        qFrameSC.put(screenout)
        event_is_set = eSC.wait()
        qFrameSC.join()
        print('qFrameSC is joined')



def face_thread():

        def detect_one_face(im):
            face_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
            gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.2, 3)
            if len(faces) == 0:
                return (0,0,0,0)
            return faces[0]

        #tracking code

        cap = cv2.VideoCapture(0)
        ret = cap.set(3,1280)
        ret = cap.set(4,720)


##        fps = cap.get(cv2.CAP_PROP_FPS)
##        print('fps of webcam')
##        print(fps)
##
##        four_cc = cap.get(cv2.CAP_PROP_FOURCC)
##        print('format')
##        print(four_cc)
##
##        buff_size = cap.get(cv2.CAP_PROP_BUFFERSIZE)
##        print('buffer size')
##        print(buff_size)

        #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        #out = cv2.VideoWriter('outputkalman&meanshift.avi',fourcc,-1, 20.0, (1280,720))

        frameCounter = 0
        #read first frame

        ret ,frame1 = cap.read()
        if ret == False:
            return


        # detect face in first frame
        c,r,w,h = detect_one_face(frame1)
        pt = (0,c+w/2,r+h/2)
        # Write track point for first frame

        frameCounter = frameCounter + 1

            # set the initial tracking window
        track_window = (c,r,w,h)
        state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
        kalman = cv2.KalmanFilter(4,2,0)
        kalman.transitionMatrix = np.array([[1., 0., 0.5, 0.],
                                            [0., 1., 0., 0.5],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]])
        kalman.measurementMatrix = 2. * np.eye(2, 4)
        kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
        kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
        kalman.errorCovPost = 1e-1 * np.eye(4, 4)
        kalman.statePost = state
        measurement = np.array([c+w/2, r+h/2], dtype='float64')


        while cap.isOpened():
            ret ,frame1 = cap.read() # return false if frame is not read. can be used to find end of the video file
            if ret == False:
                break

            #kalman prediction
            prediction = kalman.predict() #prediction
            x,y,w,h = detect_one_face(frame1) #checking measurement
            measurement = np.array([x+w/2, y+h/2], dtype='float64')

            if not (x ==0 and y==0 and w==0 and h ==0):
                 posterior = kalman.correct(measurement)
            if x ==0 and y==0 and w==0 and h ==0:
                x,y,w,h = prediction
            else:
                x,y,w,h = posterior

            print('---------------------------- x is ----------------------------------')
            print(x)
            print('--------------------------------------------------------------------')
            
            centre = (x+w/2)
            if centre<680 :
                mctrl.write(str.encode('0'))
                print('MOVING...............................1...................')
                #print(centre)
            elif centre>760 :
                mctrl.write(str.encode('2'))
                print('MOVING.........................2..................')
                #print(centre)
            else :
                mctrl.write(str.encode('1'))
                print('NOT MOVING................................')
                #print(centre)


            try:
                img2 = cv2.rectangle(frame1, (int(x),int(y)), (int(x+w),int(y+h)), 255,2)

                #q.put(pt)
                #q.join() ######blocks until kalman x values are processed in other thread

                qFrameKM.put(img2)
                event_is_set = eKM.wait()
                #print('frame combine KM is released')
                #time.sleep(0.2)

                qFrameKM.join()
                #print('qFrameKM is joined')
                #time.sleep(0.2)
            except OverflowError:
                #print('error in kalman detection--out of bounds')
                qFrameKM.put(frame1)
                event_is_set = eKM.wait()
                #print('frame combine KM is released at exception')
                qFrameKM.join()
                #print('qFrameKM is joined at exception')

            cv2.imshow("face", frame1)
            key = cv2.waitKey(1)

            #k = cv2.waitKey(1)
            frameCounter = frameCounter + 1
 



#class tfWhiteboard_thread(threading.Thread):


def tfWhiteboard_thread():
    logging.debug('Starting')
    # Number of classes the object detector can identify
    NUM_CLASSES = 1
    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `1`, we know that this corresponds to `whiteboard`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Open video file
    video = cv2.VideoCapture(0)

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    ret = video.set(3,1280)
    ret = video.set(4,720)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    #print('In TF thread about to start while loop')
    while run_code:

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)


        try:
        # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            #coordinates=[ymin, ymax, xmin, xmax, (box_to_score_map[box]*100)]
            coordinates = vis_util.return_coordinates(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=20,
                        min_score_thresh=0.999)

            #print ('coordinates of boxes new code')
            #print (coordinates)

            ymin = coordinates[0]
            ymax = coordinates[1]
            xmin = coordinates[2]
            xmax = coordinates[3]

            crop_img = frame[ymin:ymax, xmin:xmax]
##            print('TF image cropped')
            qFrameTF.put(crop_img)
##            print('TF image put')
            qFrameTF.join()


        ##########################################################
        except (UnboundLocalError, Exception):
##            print('unbound excetion at WB thread')
            black_image = np.zeros((720,1280,3),np.uint8)
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (360,640)
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2

            cv2.putText(black_image,'WHITEBOARD NOT DETECTED',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


            qFrameTF.put(black_image)
            qFrameTF.join()

        #############################################################


    logging.debug('Exiting')

##################################### Thread to combine frames ###############
#class frameCombine(threading.Thread):
def frameCombine():
    logging.debug('Starting Frame combine')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_name,fourcc, 4.25 , (1920,1080))
##    time.sleep(0.2)
    
    while run_code:
        print('Now in frame combine thread')

        ############################ Get TF frame ################################
##        
##        print('acquiring TF frame')
##        print('TF queue size')
##        print(qFrameTF.qsize())
        TF=qFrameTF.get()
        eTF.set()
##        print('eTF is set')
        resizeTF = cv2.resize(TF, (960, 540))
        eTF.clear()
        qFrameTF.task_done()
            


        ############################ Get KM Frame ##################################
        Kalman=qFrameKM.get()
##        print('acquired KM frame')
        eKM.set()
##        print('eKM is set')
        resizeKM = cv2.resize(Kalman, (960, 540))
        eKM.clear()
##        print('eKM.clear is cleared')
        qFrameKM.task_done()

 
        ############################ Get SC Frame ##################################
##        print('acquiring SC frame')
        SC=qFrameSC.get()
        eSC.set()
##        print('eSC is set')
        resizeSC = cv2.resize(SC, (960, 1080))
        SC_tophalf=resizeSC[0:540,0:960]                #[y_start: y_end, x_start:x_end]
        SC_bottomhalf=resizeSC[540:1080, 0:960]          #origin is the top left corner
        eSC.clear()
        qFrameSC.task_done()
        
        ############################# combine into a single frame ###############################
        topStack = np.hstack((SC_tophalf,resizeTF))
        bottomStack=np.hstack((SC_bottomhalf, resizeKM))
        finalFrame=np.vstack((topStack, bottomStack))
        out.write(finalFrame)
        
        currrent_time = datetime.now()
        print('current time is ........................')
        print(current_time)
    
    out.release()
    cv2.destroyAllWindows()


    logging.debug('Exiting')
    print('################################done work#######################################')
##    out.write(finalFrame)

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
)
#############################################################################################
                                        # MAIN #
#############################################################################################
if __name__ == "__main__":

    ##########################################== ARGPARSE ==#################################
    
    ap = argparse.ArgumentParser()
    ap.add_argument("time", type=int,
            help="lecture length")
    
    args = ap.parse_args()
    duration = args.time
    print('duration')
    print(duration)

    ##################################### video file name ############################
    video_name = strftime("%d-%b-%Y_%I.%M%p", gmtime())
    print('currernt time')
    print(video_name)
    output_video_name = video_name+'.avi'
    
    ############################## start and stop recording set ######################
    global start_time
    global stop_time
    global current_time
    
    
    start_time = datetime.now()
    
    stop_time = start_time + timedelta(minutes=duration)
    print('stop time is')
    print(stop_time)
##    try:
    current_time = start_time
    if current_time < stop_time:
        run_code = True
    else:
        run_code = False
    print('run_code variable is -------------------------------------------------')
    print(run_code)
##    except:
##        pass
    ##################################################################################
    
    sys.path.append("..")
    MODEL_NAME = 'inference_graph'
    CWD_PATH = os.getcwd()
    # Path to frozen detection graph .pb file.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
    
    PATH_TO_PROTOTXT = os.path.join(CWD_PATH,'deep_learning_face_detection','deploy.prototxt.txt')
    PATH_TO_CAFFE = os.path.join(CWD_PATH,'deep_learning_face_detection','res10_300x300_ssd_iter_140000.caffemodel')

    comport = 'COM3'
    mctrl = serial.Serial(comport, 9600, timeout=.1)
    
    #########################################################################################

    qFrameTF = queue.Queue()
    qFrameKM = queue.Queue()
    qFrameSC = queue.Queue()
    q = queue.Queue()

    eTF=threading.Event()
    eKM=threading.Event()
    eSC=threading.Event()

    threadLock = threading.Lock()
    threads = []


    WhiteboardThread = threading.Thread(name='tfWhiteboard', target=tfWhiteboard_thread)
    FaceThread = threading.Thread(name='kalman', target= face_thread)
    ScreenCapThread = threading.Thread(name='scren_cap', target= screen_capture)
    fCmb = threading.Thread(name='frameCombine', target=frameCombine)
##    audio= threading.Thread(name='audio', target=audio_recorder)

    WhiteboardThread.start()
    FaceThread.start()
    ScreenCapThread.start()
    fCmb.start()
##    audio.start()

    WhiteboardThread.join()
    FaceThread.join()
    fCmb.join()
    ScreenCapThread.join()
##    audio.join()



##    key = ord(getch())
##    if key == 27: #ESC
##        cap.release()
##        out.release()
##        cv2.destroyAllWindows()
##
##        exit(0)
        
    
