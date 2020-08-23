from math import sin
#from numpy import linspace, random
import numpy as np
import matplotlib.pyplot as plt





class FakeData:

	def __init__(self, seq_start, seq_end, seq_len, noise_level, anomaly_prob, anomaly_level):

		self.seq_len = seq_len
		self.seq_x = np.linspace(seq_start, seq_end, seq_len)
		self.time_steps = np.array([i for i in range(seq_len)])
		self.noise_level = noise_level
		self.seq_normal = self._get_seq_normal()

		self.anomaly_prob = anomaly_prob
		self.anomaly_level = anomaly_level		
		self.fluctuation = self._get_fluctuation()
		self.seq_anomaly = self.seq_normal + self.fluctuation

	def _get_seq_normal(self):

		return np.array([sin(x)+np.random.normal(loc=0, scale=self.noise_level) for x in self.seq_x])

	def _get_fluctuation(self):
		return np.array([np.random.normal(loc=0, scale=self.anomaly_level) 
			 			 if np.random.uniform(0,1) < self.anomaly_prob else 0 for _ in range(self.seq_len)])

	def plot(self,normal=True):
		plt.figure(figsize=(15,7))
		if normal:
			plt.plot(self.time_steps, self.seq_normal)
		else:
			plt.plot(self.time_steps, self.seq_anomaly)

	def generate_training_data(self,seq_len):

		train_x, train_y = [], []
		for i in range(self.seq_len-seq_len):
			train_x.append(self.seq_anomaly[i:seq_len+i])
			train_y.append(self.seq_anomaly[seq_len+i])
			i += 1

		return np.expand_dims(np.array(train_x), axis=-1), np.expand_dims(np.array(train_y), axis=-1)






	

