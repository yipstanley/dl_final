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
	with open('data/Piano-midi.de.pickle', 'rb') as fo:
		dataset = pickle.load(fo)

	train_data = dataset['train']
	test_data = dataset['test']

	# train_data = rectify(raw_train_data)
	# test_data = rectify(raw_test_data)
	dictionary = {}

	counter = 0
	for piece in train_data:
		unique = np.unique(piece)
		for (i, v) in enumerate(unique):
			if str(v) not in dictionary:
				dictionary[str(v)] = counter
				counter += 1

	for piece in test_data:
		unique = np.unique(piece)
		for (i, v) in enumerate(unique):
			if str(v) not in dictionary:
				dictionary[str(v)] = counter
				counter += 1

	return train_data, test_data, dictionary
