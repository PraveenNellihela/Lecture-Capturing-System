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


logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
)


############################### Screen capture thread ##############################
def screen_capture():
    current_time = datetime.now()
    logging.debug('Starting')
    if current_time<stop_time:
        print('running screen cap using if true')
        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        screenout = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)

        qFrameSC.put(screenout)
        event_is_set = eSC.wait()
        qFrameSC.join()
        print('qFrameSC is joined')

    logging.debug('Exiting')

def face_thread():
        logging.debug('Starting')
        net = cv2.dnn.readNetFromCaffe(PATH_TO_PROTOTXT, PATH_TO_CAFFE)
        
        cap = cv2.VideoCapture(0)
        ret = cap.set(3,1280)
        ret = cap.set(4,720)

        print("[INFO] starting video stream...")
        

        while run_code:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            ret ,frame = cap.read() # return false if frame is not read. can be used to find end of the video file
            if ret == False:
                break

            frame = cv2.resize(frame, (820, 400))

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 2.0,
                    (300, 300), (104.0, 177.0, 123.0))
     
            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with the
                    # prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > 0.2:

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")


                        text = "{:.2f}%".format(confidence * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                (0, 0, 255), 2)
                        cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


                        centre = (startX+endX)/2
                        if centre<380 :
                            mctrl.write(str.encode('2'))
                            print('MOVING.........1')
                            print(centre)
                        elif centre>420 :
                            mctrl.write(str.encode('0'))
                            print('MOVING.........2')
                            print(centre)
                        else :
                            mctrl.write(str.encode('1'))
                            print('NOT MOVING....center')
                            print(centre)
            
            qFrameKM.put(frame)
            event_is_set = eKM.wait()
            qFrameKM.join()
            print('qFrameKM is joined')

            cv2.imshow("face", frame)
            key = cv2.waitKey(1)


        logging.debug('Exiting')
        



#class tfWhiteboard_thread(threading.Thread):


def tfWhiteboard_thread():
    logging.debug('Starting')
    # Number of classes the object detector can identify
    NUM_CLASSES = 1
    # Load the label map.
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


    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    video = cv2.VideoCapture(0)

    ret = video.set(3,1280)
    ret = video.set(4,720)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    print('In TF thread about to start while loop')
    
    if run_code:

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

            ymin = coordinates[0]
            ymax = coordinates[1]
            xmin = coordinates[2]
            xmax = coordinates[3]

            crop_img = frame[ymin:ymax, xmin:xmax]
            print('TF image cropped')
            qFrameTF.put(crop_img)
            print('TF image put')
            qFrameTF.join()


        ##########################################################
        except (UnboundLocalError, Exception):
            print('unbound excetion at WB thread')
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

def frameCombine():
    logging.debug('Starting Frame combine')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_name,fourcc, 4.75 , (1920,1080))

    current_time = datetime.now()
    if run_code:

        ############################ Get TF frame ################################
        
        print('acquiring TF frame')
        print('TF queue size')
        print(qFrameTF.qsize())
        TF=qFrameTF.get()
        eTF.set()
        print('eTF is set')
        resizeTF = cv2.resize(TF, (960, 540))
        eTF.clear()
        qFrameTF.task_done()
            
        ############################ Get KM Frame ##################################
        Kalman=qFrameKM.get()
        print('acquired KM frame')
        eKM.set()
        print('eKM is set')
        resizeKM = cv2.resize(Kalman, (960, 540))
        eKM.clear()
        print('eKM.clear is cleared')
        qFrameKM.task_done()

 
        ############################ Get SC Frame ##################################
        print('acquiring SC frame')
        SC=qFrameSC.get()
        eSC.set()
        print('eSC is set')
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
        


    logging.debug('Exiting')
    print('################################done work#######################################')



#############################################################################################
                                        # MAIN #
#############################################################################################
if __name__ == "__main__":

    ##########################################== ARGPARSE ==#################################
    duration = int(sys.argv[1])
    
    print('duration is')
    print(duration)

    ##################################### video file name ############################
    video_name = strftime("%d-%b-%Y_%I.%M%p")
    print('currernt time')
    print(video_name)
    output_video_name = video_name+'.avi'
    
    ############################## start and stop recording set ######################
    global start_time
    global stop_time
    global current_time
    global run_code


    stop_time = datetime.now() + timedelta(minutes=duration)
    print('stop time is')
    print(stop_time)

    current_time = datetime.now()
    run_code = True    
    
    sys.path.append("..")
    MODEL_NAME = 'inference_graph'
    CWD_PATH = os.getcwd()
    # Path to frozen detection graph .pb file.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
    
    PATH_TO_PROTOTXT = os.path.join(CWD_PATH,'deep_learning_face_detection','deploy.prototxt.txt')
    PATH_TO_CAFFE = os.path.join(CWD_PATH,'deep_learning_face_detection','res10_300x300_ssd_iter_140000.caffemodel')

    comport = 'COM7'
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

    WhiteboardThread.daemon = True
    WhiteboardThread.start()
    
    FaceThread.daemon = True
    FaceThread.start()
    
    ScreenCapThread.daemon = True
    ScreenCapThread.start()

    fCmb.daemon = True
    fCmb.start()

    WhiteboardThread.join()
    FaceThread.join()
    fCmb.join()
    ScreenCapThread.join()
        

