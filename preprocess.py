import os
import tensorflow as tf
import numpy as np
import pickle

def rectify(raw_data):	
	rect_data = []
	for piece in raw_data:
		np_piece = np.zeros((len(piece), 88))
		for timestep in piece:
			np_timestep = np.asarray(timestep) - 21
			if (len(np_timestep) > 0):
				np_piece[0, np_timestep] = 1
		rect_data.append(tf.convert_to_tensor(np_piece))
	return rect_data



def get_data():
	with open('Piano-midi.de.pickle', 'rb') as fo:
		dataset = pickle.load(fo)

	raw_train_data = dataset['train']
	raw_test_data = dataset['test']

	train_data = rectify(raw_train_data)
	test_data = rectify(raw_test_data)

	return train_data, test_data
