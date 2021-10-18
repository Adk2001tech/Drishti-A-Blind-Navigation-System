
from struct import pack
from math import sin, pi
import wave
import random
from os.path import abspath
import numpy as np

from tqdm import tqdm

class Audio(object):
	"""Audio Processing"""
	def __init__(self, fps):
		super(Audio, self).__init__()
		self.fps= fps
		self.wvData = b''
		self.maxVol = 2**15-1.0
		self.freqHz = 500.0
		self.SAMPLE_RATE = 44100

	def save(self, arr):
		# Processing arr for 44.1kHZ Sample rate wave file
		for i in tqdm(range(int(arr.shape[0]//self.fps))):
		    dir_value= arr[(i*self.fps): (i+1)*self.fps, :].mean(axis=0)
		    arg= np.argmax(dir_value)
		    
		    #Left
		    if arg== 0:
		        for i in range(0, int(self.SAMPLE_RATE * 1)):
		            self.wvData += pack('h', int(1 * self.maxVol * sin(2 * pi * i * self.freqHz / self.SAMPLE_RATE)))
		            self.wvData += pack('h', int(1 * 0 * sin(2 * pi * i * self.freqHz / self.SAMPLE_RATE)))
		    #Right
		    if arg==2:
		        for i in range(0, int(self.SAMPLE_RATE * 1)):
		            self.wvData += pack('h', int(1 * 0 * sin(2 * pi * i * self.freqHz / self.SAMPLE_RATE)))
		            self.wvData += pack('h', int(1 * self.maxVol * sin(2 * pi * i * self.freqHz / self.SAMPLE_RATE)))
		    #Forward
		    if arg==1:
		        for i in range(0, int(self.SAMPLE_RATE * 1)):
		           self. wvData += pack('h', int(0.6 * self.maxVol * sin(2 * pi * i * self.freqHz / self.SAMPLE_RATE)))
		           self.wvData += pack('h', int(0.6 * self.maxVol * sin(2 * pi * i * self.freqHz / self.SAMPLE_RATE)))

		# save the bytestring as a wave file
		outputFileName = 'Audio_depth.wav'
		wv = wave.open(outputFileName, 'w')
		wv.setparams((2, 2, self.SAMPLE_RATE, 1, 'NONE', 'not compressed'))
		wv.writeframes(self.wvData)
		wv.close()
		print(f"saved {abspath(outputFileName)}")
	    








		