import os
import subprocess
from os.path import splitext
from collections import Counter
import shutil

def compare_files():
    path = os.getcwd()
    #List containing all file names + their extension in path directory
    myDir = os.listdir(path)

    #List containing all file names without their extension (see splitext doc)
    l = [splitext(filename)[0] for filename in myDir]

    #Count occurences
    a = dict(Counter(l))
    #Print files name that have same name and different extension


    src = "C:/tensorflow1/models/research/object_detection"
    dst = "C:/tensorflow1/models/research/object_detection/Videos"
    
    for k,v in a.items():
        if v > 1:
            print('combining audio/video of file: '+str(k))
            output = str(k)+'_FINAL.mkv'
                
##            source_path= 'C:\\tensorflow1\\models\\research\\object_detection\\'+output
##            destination_path= 'C:\\tensorflow1\\models\\research\\object_detection\\Video\\'+output
                        
##            print(source_path)
##            print(destination_path)
##            
            test = 'ffmpeg -i '+k+'.avi -i '+k+'.wav -c copy '+output
            print(test)
            #test = 'ffmpeg -y -i '+k+'.wav -r 30 -i '+k+'.mp4 -filter:a aresample=async=1 -c:a flac -c:v copy '+k+'_FINAL.mkv'
            cmd = test
            subprocess.call(cmd, shell=True)    
            print('successfully combined audio/video of file:  '+str(k))

            fullPath = src + "/" + output
            subprocess.Popen("mv" + " " + fullPath + " " + dst,shell=True)
##            
##            os.rename(source_path, destination)
            print('moved file')
##            print('removing temporary save of audio/video file:  '+str(k))
##            os.remove(str(path)+str(k)+'.avi')
##            os.remove(str(path)+str(k)+'.wav')
##            print('successfully removed temporary save of audio/video file:  '+str(k))
##            print(str(k)+ '  audio/video combine complete!')







          
compare_files()
