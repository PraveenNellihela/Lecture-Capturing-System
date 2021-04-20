import sys
from sys import byteorder
from array import array
from struct import pack
import argparse
import time
from time import gmtime, strftime

import pyaudio as pya
import wave



## 
### construct the argument parse and parse the arguments
##ap = argparse.ArgumentParser()
##ap.add_argument("time", type=int,
##	help="lecture length")
###args = vars(ap.parse_args())
##args = ap.parse_args()
##duration = args.time

duration = int(sys.argv[1])
print('duration is')
print(duration)


time.sleep(2)
current_time = strftime("%d-%b-%Y_%I.%M%p", gmtime())

CHUNK = 1024
FORMAT = pya.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = (60* duration) +8  #* args.time # Define using the sceduling code
WAVE_OUTPUT_FILENAME = current_time+'.wav'

p = pya.PyAudio()

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
input_device_index=0,
frames_per_buffer=CHUNK)

print ('Recording')

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel

print('done recording')
print(current_time)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
