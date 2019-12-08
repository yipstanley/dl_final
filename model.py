import tensorflow as tf
from preprocess import get_data

class Model(tf.keras.Model):
	def __init__(self, window_size, vocab_size):
		super(Model, self).__init__()

		self.vocab_size = vocab_size
		self.window_size = window_size
		self.embedding_size = 128
		self.hidden_state = 128
		self.batch_size = 50

		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size))
		self.model.add(tf.keras.layers.LSTM(self.hidden_state, return_sequences=True, input_shape=(self.window_size, self.embedding_size)))
		self.model.add(tf.keras.layers.LSTM(self.hidden_state, return_sequences=True))
		self.model.add(tf.keras.layers.Dense(self.vocab_size, activation='softmax'))

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

	def call(self, inputs, initial_state):
		ret = self.model(inputs)
		return ret

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
			probs = model.call(batch_train, None)
			loss = model.loss(probs, batch_labels)
			gradients = tape.gradient(loss, model.trainable_variables)
			model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
			print("\rBatch: {0} / {1} | Loss: {2}".format(i, batches, loss), end="\r")
	print()

def main():
	sequence_length = 50
	(train_data, test_data, dictionary) = get_data()
	(train_inputs, train_labels) = ngram(train_data, dictionary, sequence_length)
	(test_inputs, test_labels) = ngram(test_data, dictionary, sequence_length)

	model = Model(sequence_length, len(dictionary))

	train(model, train_inputs, train_labels)

if __name__ == '__main__':
	main()
