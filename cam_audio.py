# System imports
import os, argparse
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np

# Vis. imports 
import cv2
import matplotlib as mp 
import matplotlib.pyplot as plt

from tqdm import tqdm

#Custom imports
from layer import *
from model import Model
from utils import Image_pro

#Audio operation
from audio import Audio


# Argument Parser
parser = argparse.ArgumentParser(description='Drishti: A Smart Blind System-@Adk2001Tech')
parser.add_argument('-m', default='nyu', type=str, help='Model Name')
parser.add_argument('-v', default='video.mp4', type=str, help='Video Name')
args = parser.parse_args()


# DenseDepth Model(.h5)
print('Loading Model....')
if args.m== 'nyu':
	model_path= 'models/nyu.h5'
elif args.m == 'kitti':
	model_path= 'models/kitti.h5'

model= Model(path= model_path, layer= BilinearUpSampling2D)
model.build()
print('Model Loaded')

def process_img(frame):
	frame_x= cv2.resize(frame, (480, 640))
	frame_x= cv2.cvtColor(frame_x, cv2.COLOR_BGR2RGB)
	frame_x= frame_x[np.newaxis, :, :, :]
	return frame_x/255.0

# Video Capture
video_path= os.path.join(os.getcwd(), 'videos/'+ args.v)
print(video_path)
# Read frames in Loop
cam= cv2.VideoCapture(video_path)
fps = cam.get(cv2.CAP_PROP_FPS)

print(cam.get(cv2.CAP_PROP_FRAME_COUNT))
total_frame= int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

# instantiate Audio, Image_pro Classes

audio_sys= Audio(fps= int(fps))

image= Image_pro()


ret, frame= cam.read()


arr= []
for _ in tqdm(range(total_frame)):
	
	frame_x= process_img(frame)
	frame_x= model(frame_x)

	barr_values= (1.0- image.radian_prob(frame_x, n= 3))*100
	arr.append(barr_values)
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break
	#read in loop
	ret, frame= cam.read()


# After the loop release the cap object---# Destroy all the windows 
cam.release() 																					
cv2.destroyAllWindows()

# Array 
arr= np.array(arr)

print('Processing 44.1kHZ Sample rate wave file')
#Process and Save audio file
audio_sys.save(arr)

