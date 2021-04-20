from __future__ import print_function
from imutils.video import VideoStream
import argparse
import imutils
import time

import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import threading
import tensorflow as tf
"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=20,
	help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video")
args = vars(ap.parse_args())

"""
# initialize the video stream and allow the camera
# sensor to warmup

        
    


#############################################################################################



#############################################################################################

class kalman_thread(threading.Thread):

      
    def run(self):
        def detect_one_face(im):
            face_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
            gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.2, 3)
            if len(faces) == 0:
                return (0,0,0,0)
            return faces[0]

        #tracking code
        """
        cap = cv2.VideoCapture('vid5.mp4') #####    ######      #########

       
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        """  
        #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        #out = cv2.VideoWriter('outputkalman&meanshift.avi',fourcc,-1, 20.0, (640,480))
        
        frameCounter = 0
        #read first frame
        ret ,frame1 = frame
        threadLock.aquire()    


        # detect face in first frame
        c,r,w,h = detect_one_face(frame1)
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
            ##############################################################################################################################threadLock.acquire() ########### lock thread until tf thread also captures frame ############
            
            ret ,frame1 = frame          ########cap.read() # read another frame
            
            frame_expanded = np.expand_dims(frame1, axis=0)
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
                
            img2 = cv2.rectangle(frame1, (int(x),int(y)), (int(x+20),int(y+20)), 255,2)
            threadLock.release()  
            #cv2.imshow('img2',img2)
            #out.write(frame1)
                
            k = cv2.waitKey(25) & 0xff
            if k == 27:  
                break
            else:
                cv2.imwrite(chr(k)+".jpg",img2)

                
                frameCounter = frameCounter + 1
            

class tfWhiteboard_thread(threading.Thread):

      
    def run(self):
        # This is needed since the notebook is stored in the object_detection folder.
        sys.path.append("..")

        # Import utilites
        from utils import label_map_util
        from utils import visualization_utils as vis_util

        # Name of the directory containing the object detection module we're using
        MODEL_NAME = 'inference_graph'
        ############################### <----- VIDEO_NAME = 'vid5.mp4'
        
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
        threadLock.acquire() 
        # Open video file
        video = frame ################################################################# cv2.VideoCapture(PATH_TO_VIDEO)

        # Get lock to synchronize threads
        
        
        ##ret = video.set(3,1280)
        ##ret = video.set(4,720)
        
        """
        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        """
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

        ## out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))



        while(True):

            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            ret, frame2 = frame
            frame_expanded = np.expand_dims(frame2, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame2,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.9999)

            # All the results have been drawn on the frame, so it's time to display it.

            ##cv2.imshow('Object detector', frame2)

            ##########################################################################################threadLock.release()    ################################## release thread lock ##############
    
            ################################################################ crop whiteboard ####################################################


            """

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

            crop_img = frame[xmin:xmax, ymin:ymax]
            cv2.imshow("cropped", crop_img)
            
            """
            threadLock.release() 
            # All the results have been drawn on the frame, so it's time to display it.
            ## cv2.imshow('Object detector', frame)

            numpy_horizontal = np.hstack((frame2, img2))

            numpy_horizontal_concat = np.concatenate((frame2, img2), axis=1)

            cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)  
            
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
        ##threadLock.release()
        # Clean up


print("[INFO] warming up camera...")
#vs = VideoStream(usePiCamera=args["picamera"] > 0).start()######################################################
vs = cv2.VideoCapture('vid5.mp4')
#time.sleep(2.0)
 
# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv2.VideoWriter_fourcc('X' ,'V','I','D')

fps = vs.get(cv2.CAP_PROP_FPS)
print ('Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}' .format(fps))

writer = None
(h, w) = (None, None)
zeros = None


# loop over frames from the video stream
while True:
	# grab the frame from the video stream and resize it to have a
	# maximum width of 300 pixels
	frame = vs.read()
	#frame = imutils.resize(frame, width=300)
 
	# check if the writer is None
	if writer is None:
		# store the image dimensions, initialzie the video writer,
		# and construct the zeros array
		#h = frame.size(0)
		#height=frame.get(cv2.CAP_PROP_FRAME_WIDTH)
		#width=frame.get(cv2.CAP_PROP_FRAME_HEIGHT)
		#print ('Frames width, height',h, '',w)
		w=720
		h=1280
		writer = cv2.VideoWriter("output", fourcc,fps,
			(w * 2, h * 2), True)
		zeros = np.zeros((w, h), dtype="uint8")
############################################################################################

threadLock = threading.Lock()
threads = []

WhiteboardThread = tfWhiteboard_thread()   
FaceThread = kalman_thread()

WhiteboardThread.start()     
FaceThread.start()

threads.append(WhiteboardThread)
threads.append(FaceThread)



        # construct the final output frame, storing the original frame
	# at the top-left, the red channel in the top-right, the green
	# channel in the bottom-right, and the blue channel in the
	# bottom-left
"""	
output = np.zeros((h * 2, w * 2, 3), dtype="uint8")
output[0:h, 0:w] = frame
output[0:h, w:w * 2] = img2
output[h:h * 2, w:w * 2] = frame2
output[h:h * 2, 0:w] = frame
 
	# write the output frame to file
writer.write(output)
"""
for t in threads:
        t.join()



        # show the frames
cv2.imshow("Frame", frame)
cv2.imshow("Output", output)
key = cv2.waitKey(1) & 0xFF
""" 
	# if the `q` key was pressed, break from the loop
if key == ord("q"):
        break
""" 
# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
writer.release()
