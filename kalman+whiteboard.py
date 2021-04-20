import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def detect_one_face(im):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)

    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]



if __name__ == '__main__':

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    # Import utilites
    from utils import label_map_util
    from utils import visualization_utils as vis_util

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    VIDEO_NAME = 'vid5.mp4'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

    # Path to video
    PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

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

    #tracking code

    cap = cv2.VideoCapture("vid5.mp4")
    
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
##########
        width, height, channels = frame.shape
        #print (width, height)
            
#############
        
            # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.999999999)
        print ('box')
        #print (boxes[0])
        """
        margin_array=boxes[0]
        print (margin_array)

        
        ymin=np.around(margin_array[0],decimals=6)*720
        xmin=np.around(margin_array[1],decimals=6)*1280
        ymax=np.around(margin_array[2],decimals=6)*720
        xmax=np.around(margin_array[3],decimals=6)*1280

        #print ('rounded off values')
        #print (offset_height,offset_width, target_height,target_width)

        ymin=ymin.astype(int)
        xmin=xmin.astype(int)
        ymax=ymax.astype(int)
        xmax=xmax.astype(int)


        print ('ymin, xmin, ymax, xmax as int')
        print (ymin, xmin, ymax, xmax)
        """
        #crop_img = frame[ymin:ymin+ymax, xmin:xmin+xmax]
        #cv2.imshow("cropped", crop_img)

        """
        tf.image.crop_to_bounding_box(
            frame,
            offset_height,
            offset_width,
            target_height,
            target_width
        )
        """
        # All the results have been drawn on the frame, so it's time to display it.
        #cv2.imshow('Object detector', frame)

####################################
        
        ymin = boxes[0][1][0]*height
        xmin = boxes[0][1][1]*width
        ymax = boxes[0][1][2]*height
        xmax = boxes[0][1][3]*width

        print ('Top left')
        print (xmin,ymin,)
        print ('Bottom right')
        print (xmax,ymax)

        
        ymin = ymin.astype(int)
        xmin = xmin.astype(int)
        ymax = ymax.astype(int)
        xmax = xmax.astype(int)
#################################### crop whiteboard #########################################

        #crop_img = frame[xmin:xmax, ymin:ymax]
        #cv2.imshow("cropped", crop_img)
        

################################## kalman prediction #########################################
             
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
        
        img2 = cv2.rectangle(frame, (int(x),int(y)), (int(x+50),int(y+50)), 255,2)
    
        cv2.imshow('img2',img2)
        out.write(frame)
    
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

		
        frameCounter = frameCounter + 1
    
    cv2.destroyAllWindows()
    out.release()
    cap.release()



