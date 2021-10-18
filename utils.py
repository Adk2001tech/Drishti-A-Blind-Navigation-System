import os
import glob
import numpy as np

import cv2
from PIL import Image 

class Image_pro(object):
	"""Class for visualization results"""
	def __init__(self):
		super(Image_pro, self).__init__()
		self.dir= os.getcwd()

	def load(self, path, size= None, batch= False):
		img= Image.open(path)
		if size:
			img= img.resize(size)
		if batch:
			img= np.array(img)
			return img[np.newaxis, :, :, :]
		return np.array(img)

	def normalize(self, data):
		return (data - np.min(data)) / (np.max(data) - np.min(data))

	def radian_prob(self, img, n= 10, normalize= True):
		'''Arguments
		   img: Predicted Depth Gray Image Array 
		   n: n-split'''
		if img.ndim == 4:
			img= img[0, :, :, 0]
		elif img.ndim == 3:
			img= img[0, :, :]
		# milliradian seen/covered by normal human EYE
		milliradian= 2100
		split_sp= milliradian//n
		depth_arr= []
		img= cv2.resize(img, (milliradian, img.shape[-1]))
		for n_split in range(1,n+1):
		    depth_arr.append(img[:, (n_split-1)* split_sp: (n_split)*split_sp].sum())
		if normalize:
			return self.normalize(depth_arr)
		else:
			return np.array(depth_arr) 


			


		