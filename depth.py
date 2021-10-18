# System imports
import os, argparse, glob
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from tqdm import tqdm

# Vis. imports 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import seaborn as sns
sns.set_theme(style="whitegrid")

#Custom imports
from layer import *
from model import Model
from utils import Image_pro


# Argument Parser
parser = argparse.ArgumentParser(description='Drishti: A Smart Blind System-@Adk2001Tech')
parser.add_argument('-m', default='nyu', type=str, help='Model Name')
parser.add_argument('-i', default='images', type=str, help='Image folder')
parser.add_argument('-s', type=int, default=5, help='n-split frame')
args = parser.parse_args()

if args.s==5:
	direction =['ext-left', 'left', 'forward', 'right', 'ext-right']
elif args.s==3:
	direction =['left', 'forward', 'right']


#DenseDepth Model(.h5)
print('Loading Model....')
if args.m== 'nyu':
	model_path= 'models/nyu.h5'
elif args.m == 'kitti':
	model_path= 'models/kitti.h5'

model= Model(path= model_path, layer= BilinearUpSampling2D)
model.build()
print('Model Loaded')

print(glob.glob(args.i+ '/*jpg'))
image= Image_pro()

for img in tqdm(glob.glob(args.i+ '/*jpg')):
	# make a Figure and attach it to a canvas.
	fig = Figure(figsize=(10, 10), dpi=100)
	canvas = FigureCanvasAgg(fig)

	path= os.path.join(os.getcwd(), img)
	l= len(args.i)

	img_x= image.load(path= path, size=(640, 480), batch= True)
	#plt.imshow(img_x[0])

	result_img= model(img_x/255.0)
	bar_values= (1.0-image.radian_prob(result_img, n= args.s))*100
	result_img = cv2.resize(result_img[0, :, :, 0],(640, 480))
	result_img= image.normalize(result_img)*255.0
	save_path= img[:l]+ '/depth_' +img[l+1:]
 
    # Do some plotting here
	ax = fig.add_subplot(111)
	ax.bar(direction, bar_values)
	# Retrieve a view on the renderer buffer
	canvas.draw()
	buf = canvas.buffer_rgba()
	# convert to a NumPy array
	bar_arr = np.asarray(buf)
	bar_arr = cv2.resize(bar_arr,(640, 480))[:,:, :-1]


	hstack= np.hstack(( cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB), bar_arr ))
	result_img= Image.fromarray(np.uint8(hstack))
	result_img.save(save_path)
