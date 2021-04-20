from sys import byteorder
from array import array
from struct import pack

import pyaudio as pya
import wave

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
