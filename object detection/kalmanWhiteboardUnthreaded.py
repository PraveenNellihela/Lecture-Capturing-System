import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imutils.video import FileVideoStream
from utils import label_map_util
from utils import visualization_utils as vis_util
import timeit
import time
from imutils.video import FPS

if __name__ == '__main__':
    start_time = time.time()
    print('Main thread started.... Warming up....')
    
    sys.path.append("..")
    MODEL_NAME = 'inference_graph'
    VIDEO_NAME = 'vid5.mp4'
    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
    PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)
    NUM_CLASSES = 1

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
    
###############################################################################################

    video = cv2.VideoCapture(PATH_TO_VIDEO)

    
    while(1):


        print('Tensorflow detection for one frame --')

        ########################### TENSORFLOW #######################
       
 
        ret ,frame = video.read() 
        frame_expanded = np.expand_dims(frame, axis=0)       

        # detection
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.999)
        
        #print ('coordinates of boxes new code')
        #print (coordinates)
            
##        try:
##            ymin = coordinates[0]
##            ymax = coordinates[1]
##            xmin = coordinates[2]
##            xmax = coordinates[3]
##        except IndexError:
##            pass
        
        cv2.imshow('object detector', frame)
        ########################## END TENSORFLOW ###################
        cv2.waitKey(1)
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps())) 
        print("--- %s seconds for processing one tensorflow detection frame ---" % (time.time() - start_time))

        ########################## kalman prediction #########################################
             
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('outputkalman&meanshift.avi',fourcc,-1, 30.0, (1280,720))

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
        cv2.waitKey(1)
        		
        frameCounter = frameCounter + 1
        print("--- %s seconds for processing 1 loop ---" % (time.time() - start_time))

print("--- %s seconds for processing entire video ---" % (time.time() - start_time))
cv2.destroyAllWindows()
out.release()
cap.release()



