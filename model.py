import tensorflow as tf
from tensorflow.keras.models import load_model

class Model(object):
	""" Model class for loading NYU Depth V2 model"""
	def __init__(self, path, layer):
		super(Model, self).__init__()
		'''Arguments
		   path: Path for .h5 Models
		   		 (1. NYU Depth V2), (2. KITTI)
		   layer:  Up-sampling Conv2D layer (defined in layer.py)'''

		self.path = path
		self.BilinearUpSampling2D= layer
	def build(self):

		# Custom object needed for inference and training
		custom_objects = {'BilinearUpSampling2D': self.BilinearUpSampling2D, 'depth_loss_function': None}
		# Load model into GPU / CPU
		self.model = load_model(self.path, custom_objects=custom_objects, compile=False)
		
	def __call__(self, x):
		return self.model.predict(x)
