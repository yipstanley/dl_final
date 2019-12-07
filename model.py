import tensorflow as tf
from preprocess import get_data

class Model(tf.keras.Model):
    def __init__(self, window_size, vocab_size):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_size = 80


def ngram(input_arr, dictionary, n):
    end = input_arr.shape[0] - n
    inputs = []
    labels = []
    for i in range(0, end, n):
        window = []
        label = []
        for j in range(i, i + n):
            window.append(dictionary[input_arr[j]])
            label.append(dictionary[input_arr[j+1]])
        inputs.append(window)
        labels.append(label)
    inputs = tf.reshape(tf.convert_to_tensor(inputs), (-1, n))
    labels = tf.reshape(tf.convert_to_tensor(labels), (-1, n))
    return (inputs, labels)

def main():
    (train_data, test_data, dictionary) = get_data()