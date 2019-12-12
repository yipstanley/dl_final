import tensorflow as tf
#from midiutil import MIDIFile
import os
import sys
from preprocess import get_data

class Model(tf.keras.Model):
	def __init__(self, window_size, vocab_size):
		super(Model, self).__init__()

		self.vocab_size = vocab_size
		self.window_size = window_size
		self.embedding_size = 128
		self.hidden_state = 128
		self.batch_size = 50

#		self.model = tf.keras.Sequential()
#		self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size))
#		self.model.add(tf.keras.layers.LSTM(self.hidden_state, return_sequences=True, input_shape=(self.window_size, self.embedding_size)))
#		self.model.add(tf.keras.layers.LSTM(self.hidden_state, return_sequences=True))
#		self.model.add(tf.keras.layers.Dense(self.vocab_size, activation='softmax'))

		self.E = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size)
		self.lstm_1 = tf.keras.layers.LSTM(self.hidden_state, return_sequences=True, input_shape=(self.window_size, self.embedding_size))
		self.lstm_2 = tf.keras.layers.LSTM(self.hidden_state, return_sequences=True, return_state=True)
		self.dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

	def call(self, inputs, initial_state):
		#ret = self.model(inputs)
		emb = self.E(inputs)
		l1 = self.lstm_1(emb, initial_state=initial_state)
		l2, state_h, state_c = self.lstm_2(l1)
		ff = self.dense(l2)
		return ff, state_h, state_c

	def loss(self, probs, labels):
		return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

def ngram(input_arr, dictionary, n):
	"""end = len(input_arr) - n
	inputs = []
	labels = []
	for i in range(0, end, n):
		window = []
		label = []
		for j in range(i, i + n):
			window.append(dictionary[str(input_arr[j])])
			label.append(dictionary[str(input_arr[j+1])])
		inputs.append(window)
		labels.append(label)
	inputs = tf.reshape(tf.convert_to_tensor(inputs), (-1, n))
	labels = tf.reshape(tf.convert_to_tensor(labels), (-1, n))
	return (inputs, labels)"""
	inputs = []
	labels = []
	for i in range(len(input_arr)):
		end = len(input_arr[i]) - n
		for j in range(i, end):
			window = []
			label = []
			for k in range(j, j + n):
				window.append(dictionary[str(input_arr[i][k])])
				label.append(dictionary[str(input_arr[i][k+1])])
			inputs.append(window)
			labels.append(label)
	
	inputs = tf.reshape(tf.convert_to_tensor(inputs), (-1, n))
	labels = tf.reshape(tf.convert_to_tensor(labels), (-1, n))
	return (inputs, labels)

def train(model, train_inputs, train_labels):
	batches = int(len(train_inputs) / model.batch_size)
	for i in range(batches):
		with tf.GradientTape() as tape:
			batch_train = train_inputs[i * model.batch_size: (i + 1) * model.batch_size]
			batch_labels = train_labels[i * model.batch_size: (i + 1) * model.batch_size]
			probs, state_h, state_c = model.call(batch_train, None)
			loss = model.loss(probs, batch_labels)
			gradients = tape.gradient(loss, model.trainable_variables)
			model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
			print("\rBatch: {0} / {1} | Loss: {2}".format(i, batches, loss), end="\r")
	print()

def generate_piece(model, input, beats, vocab, tempo=60):
	reverse_vocab = {idx:word for word, idx in vocab.items()}
	initial_state = None
	next_input = [input]
	piece = [reverse_vocab[input[0].numpy()]]
	for beat in range(beats):
		timestep, state_h, state_c = model(tf.convert_to_tensor(next_input), initial_state)
		initial_state = state_h, state_c

		next_input = tf.argmax(timestep, axis=2)
		piece.append(reverse_vocab[tf.argmax(timestep[0][0]).numpy()])
	
	return piece


def test(model, test_inputs, vocab):
	for piece in test_inputs:
		input = []
		for t in range(50):
			input.append(vocab[str(piece[t])])
		input = tf.convert_to_tensor(input)

		beats = 80
		track = 0
		tempo = 60

		generated = generate_piece(model, input, beats, vocab, tempo)
		midi = MIDIFile(1)
		midi.addTempo(track, beats, tempo)

		for i, pitch in enumerate(generated):
			print(pitch)
			return


def main():
	if len(sys.argv) < 2:
		print("USAGE: python model.py <TEST/TRAIN> [r]")
		exit()
	
	sequence_length = 50
	(train_data, test_data, dictionary) = get_data()
	(train_inputs, train_labels) = ngram(train_data, dictionary, sequence_length)

	model = Model(sequence_length, len(dictionary))
	num_epochs = 10

	restore_checkpoint = len(sys.argv) == 3 and sys.argv[2] == "r"
	output_dir = "./output"
	checkpoint_dir = "./checkpoints"
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
	manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	if restore_checkpoint:
		checkpoint.restore(manager.latest_checkpoint)

	if sys.argv[1] == "train":
		for epoch in range(0, num_epochs):
			print("Epoch {0}".format(epoch + 1))
			train(model, train_inputs, train_labels)
			manager.save()
	elif sys.argv[1] == "test":
		test(model, test_data, dictionary)

if __name__ == '__main__':
	main()
