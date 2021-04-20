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
from tkinter import *



##from sys import byteorder
from array import array
from struct import pack

import pyaudio as pya
import wave

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import label_map_util
from utils import visualization_utils as vis_util
from msvcrt import getch
from PIL import ImageGrab
from imutils.video import VideoStream

global running 


logging.basicConfig(level=logging.DEBUG,
    format='[%(levelname)s] (%(threadName)-10s) %(message)s',)

############################### Audio capture thread ##############################
def audio_recorder():
    while running != False:
        logging.debug('Starting')
        CHUNK = 1024
        FORMAT = pya.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = 10 # Define using the sceduling code
        WAVE_OUTPUT_FILENAME = 'test.wav'

        p = pya.PyAudio()

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
        input_device_index=0,
        frames_per_buffer=CHUNK)

        print ('Recording')

        frames = []
        
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

        ################### stop recording ##############################
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data) # 2 bytes(16 bits) per channel

        print('done recording')

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()



############################### Screen capture thread ##############################
def screen_capture():
    logging.debug('Starting')
    while running != False:
        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        screenout = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)

        qFrameSC.put(screenout)
        event_is_set = eSC.wait()
        qFrameSC.join()
        print('qFrameSC is joined')




    logging.debug('Exiting')
#class face_thread(threading.Thread):

def face_thread():
        logging.debug('Starting')
        
        net = cv2.dnn.readNetFromCaffe(PATH_TO_PROTOTXT, PATH_TO_CAFFE)
##        vs = VideoStream(src=0).start()


        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        cap = cv2.VideoCapture(0)
        ret = cap.set(3,1280)
        ret = cap.set(4,720)

        print("[INFO] starting video stream...")
        time.sleep(2.0)
        

        while (cap.isOpened()) and (running != False):
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            ret ,frame = cap.read() # return false if frame is not read. can be used to find end of the video file
            if ret == False:
                break

            frame = cv2.resize(frame, (820, 400))

            
##            frame = vs.read()
##            frame = imutils.resize(frame, width=400)
     
            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
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
                    if confidence > 0.5:
    ##                            qFrameKM.put(frame)
    ##                            print('qFrameKM is sent to framecombine')
    ##                            event_is_set = eKM.wait()
    ##                            print('confidence is less than preset')
    ##                            qFrameKM.join()
    ##                            print('qFrameKM is joined')
                        # compute the (x, y)-coordinates of the bounding box for the
                        # object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")


##                        try:
                                # draw the bounding box of the face along with the associated
                                # probability
                        text = "{:.2f}%".format(confidence * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                (0, 0, 255), 2)
                        cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


                        centre = (startX+startY)/2
                        if centre<180 :
                            mctrl.write(str.encode('0'))
                            print('MOVING................................................1.........................................')
                            print(centre)
                        elif centre>220 :
                            mctrl.write(str.encode('2'))
                            print('MOVING................................................2.........................................')
                            print(centre)
                        else :
                            mctrl.write(str.encode('1'))
                            print('NOT MOVING................................................center................')
                            print(centre)
            
            qFrameKM.put(frame)
            event_is_set = eKM.wait()
            qFrameKM.join()
            print('qFrameKM is joined')
                                #time.sleep(0.2)
##                        except OverflowError:
##                                print('error in kalman detection--out of bounds')
##                                qFrameKM.put(frame)##############################################################################################################
##                                event_is_set = eKM.wait()
##                                #print('frame combine KM is released at exception')
##                                qFrameKM.join()
##                                #print('qFrameKM is joined at exception')
            
##                    else:
##                        qFrameKM.put(frame)
##                        event_is_set = eKM.wait()
##                        print('face thread now in else statement.........................................')
##                        #time.sleep(0.2)
##                        qFrameKM.join()

                            

            # show the output frame
            cv2.imshow("face", frame)
            key = cv2.waitKey(1)


            logging.debug('Exiting')



#class tfWhiteboard_thread(threading.Thread):


def tfWhiteboard_thread():
        logging.debug('Starting')
        # Number of classes the object detector can identify
        NUM_CLASSES = 1
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
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

        while(video.isOpened()) and (running != False):

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

                qFrameTF.put(crop_img)
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
#class frameCombine(threading.Thread):
def frameCombine():
    logging.debug('Starting')
    #time.sleep(0.2)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('outputNEW.avi',fourcc, 4.25 , (1920,1080))
    #time.sleep(0.2)
    while running != False:

        ############################ Get TF frame ################################
        print('acquiring TF frame')
        TF=qFrameTF.get()
        eTF.set()
        resizeTF = cv2.resize(TF, (960, 540))
        eTF.clear()

        
            

        ############################ Get KM Frame ##################################
        Kalman=qFrameKM.get()
        print('acquired KM frame')
        eKM.set()
        print('eKM is set')
        resizeKM = cv2.resize(Kalman, (960, 540))
        eKM.clear()
        print('eKM.clear is cleared')
        
        ############################ Get SC Frame ##################################
        print('acquiring SC frame')
        SC=qFrameSC.get()
        eSC.set()
        print('eSC is set')
        resizeSC = cv2.resize(SC, (960, 1080))
        SC_tophalf=resizeSC[0:540,0:960]                #[y_start: y_end, x_start:x_end]
        SC_bottomhalf=resizeSC[540:1080, 0:960]          #origin is the top left corner
        eSC.clear()
##        cv2.imshow("TF", SC_tophalf)              #
##        cv2.waitKey(1)                          #

        ###############################################################################

##        black_image2 = np.zeros((720,1280,3),np.uint8)
##        font                   = cv2.FONT_HERSHEY_SIMPLEX
##        bottomLeftCornerOfText = (360,640)
##        fontScale              = 1
##        fontColor              = (255,255,255)
##        lineType               = 2
##
##        cv2.putText(black_image2,'makeup image',
##            bottomLeftCornerOfText,
##            font,
##            fontScale,
##            fontColor,
##            lineType)
##
##        black_image2Rz=cv2.resize(black_image2, (960, 540))

        ############################# combine into a single frame ###############################
        topStack = np.hstack((SC_tophalf,resizeTF))
        bottomStack=np.hstack((SC_bottomhalf, resizeKM))
        finalFrame=np.vstack((topStack, bottomStack))
        out.write(finalFrame)

##        cv2.imshow('imgc',finalFrame)
##        cv2.waitKey(1)

        qFrameKM.task_done()
        qFrameTF.task_done()
        qFrameSC.task_done()


    logging.debug('Exiting')
    print('###################################################################################')
    out.write(finalFrame)


#############################################################################################
                                    # MAIN #
#############################################################################################
if __name__ == "__main__":
    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")
    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
    PATH_TO_PROTOTXT = os.path.join(CWD_PATH,'deep_learning_face_detection','deploy.prototxt.txt')
    PATH_TO_CAFFE = os.path.join(CWD_PATH,'deep_learning_face_detection','res10_300x300_ssd_iter_140000.caffemodel')

    mctrl = serial.Serial('COM3', 9600, timeout=.1)
    running = True

    ##################################### whiteboard update timeout ############################

    
    #############################################################################################

    

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
    audioRecorder = threading.Thread(name='audio_thread', target=audio_recorder)

    WhiteboardThread.start()
    FaceThread.start()
    ScreenCapThread.start()
    fCmb.start()
    audioRecorder.start()

    WhiteboardThread.join()
    FaceThread.join()
    fCmb.join()
    ScreenCapThread.join()
    audioRecorder.join()
#############################################################################################################
    #############################################################################################################
    #############################################################################################################

##    print ('Normal recording\nMuxing')
##    cmd = "ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video.avi -pix_fmt yuv420p " + filename + ".avi"
##    subprocess.call(cmd, shell=True)

##    running = False
#############################################################################################################
   #############################################################################################################
    #############################################################################################################

# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
##    class Window(Frame):
##
##        # Define settings upon initialization. Here you can specify
##        def __init__(self, master=None):
##            
##            # parameters that you want to send through the Frame class. 
##            Frame.__init__(self, master)   
##
##            #reference to the master widget, which is the tk window                 
##            self.master = master
##
##            #with that, we want to then run init_window, which doesn't yet exist
##            self.init_window()
##
##        #Creation of init_window
##        def init_window(self):
##
##            # changing the title of our master widget      
##            self.master.title("GUI")
##
##            # allowing the widget to take the full space of the root window
##            self.pack(fill=BOTH, expand=1)
##
##            # creating a button instance
##            quitButton = Button(self, text="Exit",command=self.client_exit)
##
##            # placing the button on my window
##            quitButton.place(x=0, y=0)
##
##           
##
##        def client_exit(self):
##            exit()
##
##    # root window created. Here, that would be the only window, but
##    # you can later have windows within windows.
##    root = Tk()
##
##    root.geometry("400x300")
##
##    #creation of an instance
##    app = Window(root)
##
##    root.mainloop() 


        
    
