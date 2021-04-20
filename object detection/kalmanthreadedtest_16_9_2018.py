import os
import sys
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import threading
import argparse
#from threading import Thread
import tensorflow as tf
import queue
from imutils.video import FileVideoStream

class kalman_thread(threading.Thread):

      
    def run(self):
        def detect_one_face(im):
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.2, 3)
            if len(faces) == 0:
                return (0,0,0,0)
            return faces[0]

        #tracking code

        cap = cv2.VideoCapture('vid3.mp4')
       
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('outputkalman&meanshift.avi',fourcc,-1, 12.0, (1280,720))

        frameCounter = 0
        #read first frame
        ret ,frame1 = cap.read()
            
        #frameCounter = frameCounter + 1

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
        
            
        while(1):
            ret ,frame1 = cap.read() # read another frame
            threadLock.acquire() ########### lock thread until tf thread also captures frame ############
            #frame_expanded = np.expand_dims(frame1, axis=0)
            #if ret == False:
             #   break
                

                
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
                
            img2 = cv2.rectangle(frame1, (int(x),int(y)), (int(w),int(h)), 255,2)
            
            q.put(pt)
            
            #cv2.moveWindow('img2',720,720);
            cv2.imshow('img2',img2)

            q.join() ######blocks until kalman x values are processed in other thread
            #cv2.moveWindow('img2',720,720);
            #out.write(frame1)

            k = cv2.waitKey(25) & 0xff
            if k == 27:
                threadLock.release()  
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
        VIDEO_NAME = 'vid3.mp4'

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

        # Open video file
        video = cv2.VideoCapture(PATH_TO_VIDEO)

        

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
         
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))



        while(video.isOpened()):

            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            ret, frame = video.read()
            frame_expanded = np.expand_dims(frame, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            """ 
            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.999)
            """
            ################################## crop and resize whiteboard #############################################################

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

            print ('coordinates of boxes new code')
            print (coordinates)
            
            try:
                ymin = coordinates[0]
                ymax = coordinates[1]
                xmin = coordinates[2]
                xmax = coordinates[3]
            except IndexError:
                pass
            
            
            kalman_x = q.get();
           
            print('kalman x =')
            print(kalman_x)
            q.task_done()
            #crop and resize
            crop_img = frame[ymin:ymax, xmin:xmax]
            
            resize = cv2.resize(crop_img, (640, 360))
            cv2.imshow("resized", resize)

           


            ################################################################################################
            threadLock.release()    ################################## release thread lock ##############
                # Write the frame into the file 'output.avi'
            out.write(frame)
            
            
            """
            # All the results have been drawn on the frame, so it's time to display it.
            #cv2.imshow('Object detector', frame)
            #cv2.moveWindow('frame', 0, 0);
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                threadLock.release()  
                break
        # Clean up

            """
#class frameCombine(threading.Thread):
"""    
class frameCombine(threading.Thread):
	def __init__(self, queueSize=128):      #can add path here in future
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture('vid3.mp4')
		self.stopped = False
 
		# initialize the queue used to store frames read from
		# the video file
		self.Q = queue.Queue(maxsize=queueSize)
		
	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
 
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
 
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
 
				# add the frame to the queue
				self.Q.put(frame)

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
"""
#############################################################################################

print("[INFO] starting video file thread...")
#fvs = FileVideoStream('vid3.mp4').start()
#time.sleep(1.0)

q = queue.Queue()


threadLock = threading.Lock()
threads = []


WhiteboardThread = tfWhiteboard_thread()   
FaceThread = kalman_thread()
#frameCombineThread = frameCombine()

WhiteboardThread.start()     
FaceThread.start()
#frameCombineThread.start()



threads.append(WhiteboardThread)
threads.append(FaceThread)
#threads.append(frameCombineThread)


for t in threads:
    t.join()
"""
k = cv2.waitKey(25) & 0xff
if k == 27:
    threadLock.release()  
    break
else:
    cv2.imwrite(chr(k)+".jpg",img2)

"""

out.release()
cap.release()
video.release()
cv2.destroyAllWindows()
