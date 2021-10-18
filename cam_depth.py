# System imports
import os, argparse
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np

# Vis. imports 
import cv2
import matplotlib as mp 
import matplotlib.pyplot as plt

#Custom imports
from layer import *
from model import Model
from utils import Image_pro


# Argument Parser
parser = argparse.ArgumentParser(description='Drishti: A Smart Blind System-@Adk2001Tech')
parser.add_argument('-m', default='nyu', type=str, help='Model Name')
parser.add_argument('-v', default='video.mp4', type=str, help='Video Name')
parser.add_argument('-s', type=int, default=5, help='n-split frame')
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

image= Image_pro()

def process_img(frame):
	frame_x= cv2.resize(frame, (480, 640))
	frame_x= cv2.cvtColor(frame_x, cv2.COLOR_BGR2RGB)
	frame_x= frame_x[np.newaxis, :, :, :]
	return frame_x/255.0

# Colorize the graph based on likeability:
data = [2, 8, 3, 14, 6]
likeability_scores = np.array(data)
 
data_normalizer = mp.colors.Normalize()
color_map = mp.colors.LinearSegmentedColormap(
    "my_map",
    {
        "green": [(0, 1.0, 1.0),
                (1.0, .5, .5)],
        "red": [(0, 0.8, 0.5),
                  (1.0, 0, 0)],
        "blue": [(0, 0.3, 0.8),
                 (1.0, 0, 0)]
    })

n= args.s
# Initialize plot.
fig, ax = plt.subplots()
ax.set_title('Drishti: A Smart Blind System')
ax.set_xlabel('Minor-Radian (-60* -- +60*)')
ax.set_ylabel('Probability (per Direction)')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
#bins= n
if n==5:
	direction =['ext-left', 'left', 'forward', 'right', 'ext-right']
elif n==3:
	direction =['left', 'forward', 'right']

lineGray = ax.bar(direction,  np.zeros(n), color=color_map(data_normalizer(likeability_scores)))
ax.set_ylim(0, 100)
plt.ion()
plt.show()


video_path= os.path.join(os.getcwd(), 'videos/'+ args.v)
print(video_path)
# Read frames in Loop
cam= cv2.VideoCapture(video_path)
cv2.namedWindow('Original Frame', cv2.WINDOW_NORMAL)

ret, frame= cam.read()

while ret:
	
	frame_x= process_img(frame)
	frame_x= model(frame_x)
	display= cv2.resize(frame_x[0, :, :, 0],(640, 480))
	display= image.normalize(display)
	display= cv2.cvtColor(display, cv2.COLOR_GRAY2RGB);frame= cv2.resize(frame,(640, 480))
	

	display = np.hstack((frame/255.0, display))


	cv2.imshow('Original Frame',display)

	barr_values= (1.0- image.radian_prob(frame_x, n= n))*100
	for line, h in zip(lineGray, barr_values):
		line.set_height(h)
	fig.canvas.draw()


	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break
	#read in loop
	ret, frame= cam.read()


# After the loop release the cap object---# Destroy all the windows 
cam.release() 																					
cv2.destroyAllWindows()
