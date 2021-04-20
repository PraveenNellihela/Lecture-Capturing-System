import os
import sys
import cv2
import numpy as np
import threading
import tensorflow as tf
import queue
import logging
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import label_map_util
from utils import visualization_utils as vis_util
from msvcrt import getch
from PIL import ImageGrab

############################### Screen capture thread ##############################
def screen_capture():
    while True:
        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        screenout = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)

        qFrameSC.put(screenout)
        event_is_set = eSC.wait()
        qFrameSC.join()
        print('qFrameSC is joined')
    

#class kalman_thread(threading.Thread):

def kalman_thread():
        def detect_one_face(im):
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
        kalman.transitionMatrix = np.array([[2., 0., 1., 0.],
                                            [0., 2., 0., 1.],
                                            [0., 0., 2., 0.],
                                            [0., 0., 0., 2.]])
        kalman.measurementMatrix = 1. * np.eye(2, 4)
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


            pt = (frameCounter,x+w/2,y+h/2)
            print (pt)
            try:
                img2 = cv2.rectangle(frame1, (int(x),int(y)), (int(x+w),int(y+h)), 255,2)

                #q.put(pt)
                #q.join() ######blocks until kalman x values are processed in other thread

                qFrameKM.put(img2)
                event_is_set = eKM.wait()
                #print('frame combine KM is released')
                #time.sleep(0.2)

                qFrameKM.join()
                print('qFrameKM is joined')
                #time.sleep(0.2)
            except OverflowError:
                print('error in kalman detection--out of bounds')
                qFrameKM.put(frame1)
                event_is_set = eKM.wait()
                #print('frame combine KM is released at exception')
                qFrameKM.join()
                #print('qFrameKM is joined at exception')



            #k = cv2.waitKey(1)
            frameCounter = frameCounter + 1


#class tfWhiteboard_thread(threading.Thread):


def tfWhiteboard_thread():
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

        while(video.isOpened()):

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



##################################### Thread to combine frames ###############
#class frameCombine(threading.Thread):
def frameCombine():
    logging.debug('Starting')
    #time.sleep(0.2)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('outpy2.avi',fourcc, 4.25 , (1920,540))
    #time.sleep(0.2)
    while True:

        ############################ Get TF frame ################################
        print('acquiring TF frame')
        TF=qFrameTF.get()
        eTF.set()
        resizeTF = cv2.resize(TF, (960, 540))

##        cv2.imshow("TF", resizeTF)              #
##        cv2.waitKey(1)                          #

##        blur = cv2.GaussianBlur(resize,(5,5),0)
##        resizeTF = cv2.addWeighted(blur,1.5,resize,-0.5,0)
        eTF.clear()

        ############################ Get KM Frame ##################################
        Kalman=qFrameKM.get()
        print('acquired KM frame')
        eKM.set()
        #print('eKM is set')
        resizeKM = cv2.resize(Kalman, (960, 540))
        eKM.clear()

        ############################ Get SC Frame ##################################
        print('acquiring SC frame')
        SC=qFrameSC.get()
        eSC.set()
        print('eSC is set')
        resizeSC = cv2.resize(SC, (960, 540))
        eSC.clear()
        cv2.imshow("TF", resizeSC)              #
        cv2.waitKey(1)                          #

       
        
        ############################# combine into a single frame ###############################
        both = np.hstack((resizeTF,resizeKM))
        out.write(both)
       
##        cv2.imshow('imgc',resizeSC)
##        cv2.waitKey(1)

        qFrameKM.task_done()
        qFrameTF.task_done()
        qFrameSC.task_done()


    logging.debug('Exiting')
    print('###################################################################################')
    #out.write(both)

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
)
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
    FaceThread = threading.Thread(name='kalman', target= kalman_thread)
    ScreenCapThread = threading.Thread(name='scren_cap', target= screen_capture)
    fCmb = threading.Thread(name='frameCombine', target=frameCombine)


    WhiteboardThread.start()
    FaceThread.start()
    ScreenCapThread.start()
    fCmb.start()

    WhiteboardThread.join()
    FaceThread.join()
    fCmb.join()
    ScreenCapThread.join()

    key = ord(getch())
    if key == 27: #ESC
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        exit(0)
