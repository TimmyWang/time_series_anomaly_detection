import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv1D, Dense
#from keras.layers import LSTM, TimeDistributed
from keras.layers import Flatten#, Reshape, Concatenate
import numpy as np
import matplotlib.pyplot as plt





def get_model(seq_len):

	seq_input = Input(shape=(seq_len,1))

	h_layer = Conv1D(3, 10,activation="relu")(seq_input)
	h_layer = Flatten()(h_layer)
	h_layer = Dense(5,activation="relu")(h_layer)

	output = Dense(1)(h_layer)

	return Model(seq_input, output)


class AnomalyDetector:

	def __init__(self, model, x, y):

		self.model = model
		self.x = x
		self.y = y		
		self.time_steps = np.array([i for i in range(len(x))])
		self.values = np.squeeze(y)
		self.loss = self._calculate_loss()

	def _calculate_loss(self):

		loss = []		
		for i, x in enumerate(self.x):
			inputt = np.expand_dims(x,axis=0)
			output = np.expand_dims(self.y[i],axis=-1)
			a_loss = self.model.evaluate(inputt,output,verbose=0)
			loss.append(a_loss)

		return loss

	def plot(self,threshold,size):

		col = ['r' if l > threshold else 'b' for l in self.loss]
		plt.figure(figsize=(15,7))
		plt.scatter(self.time_steps, self.values, c=col,alpha=0.5, s=size)





 



		


