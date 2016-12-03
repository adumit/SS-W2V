from collections import deque
import pickle
import numpy as np
import math
import random
from datetime import datetime
import tensorflow as tf
import sys

batch_size = 1000
embedding_size = 200  # Dimension of the embedding vector.
context_window = 5    # How many words to consider left and right.
num_negs = 2          # How many times to reuse an input to generate a label.
num_pos = 48
vocabulary_size = 20000

if len(sys.argv) > 1:
    sys_path = sys.argv[1]
else:
    sys_path = ""
print(sys_path)


# ------------- Data Generators -------------
def gen_training_examples(sentence, pos_sentence, window):
    """
    :param sentence: A single sentence of numbers that map to the vocabulary
    :param window: Number of words to look to the left and right
    :param neg_s: Number of negative examples to generate per positive example
    :return: List of positive examples and negative examples
    """
    # 0-pad the sentence for window searching
    training_sentence = [0] * window + sentence + [0] * window
    training_examples = []
    for i in range(len(training_sentence)):
        if training_sentence[i] == 0:
            continue
        # Following the advice of Mikolov et al., subsample frequent words by
        # Randomly discarding training examples with the following probability:
        if random.random() > 1 - math.sqrt(
                        0.00001/frequencies[num_to_word[training_sentence[i]]]):
            continue
        # A positive training example is a tuple of (correct word, context)
        # Where context is a list of numbers that indicate the words to the left
        # and right in the desired window
        training_examples.append(
            (training_sentence[i], pos_sentence[i],
             training_sentence[i-window:i] + training_sentence[i+1:i+1+window]))
    return training_examples


def generate_batch(batch_size, context_window):
    global training_data
    batch = []
    cur_size = 0
    while cur_size < batch_size:
        sample = training_data.popleft()
        examples = gen_training_examples(sample[0], sample[1], context_window)
        cur_size += len(examples)
        batch += examples
    # If this generated too many examples, remove negative examples until we
    # get to the correct number
    return batch


def weight_variable(shape, name):
    """
    Initialize weights
    :param shape: shape of weights
    :param name: name of the variable
    :return: a tensor variable for weights with initial values
    """
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape, name):
    """
    Initialize biases
    :param shape: shape of biases
    :param name: name of the bias variable
    :return: a tensor variable for biases with initial values
    """

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    b = tf.Variable(tf.constant(0.1, shape=shape), name=name)
    return b


# ------------- Network Building -------------
train_inputs = tf.placeholder(tf.int32, shape=[batch_size, vocabulary_size])
train_words = tf.placeholder(tf.int32, shape=[batch_size, 1])
train_pos = tf.placeholder(tf.int32, shape=[batch_size, num_pos])

#
embeddings = weight_variable(shape=[vocabulary_size, embedding_size],
                             name="embeddings")
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# First hidden layer
weight_1 = weight_variable(shape=[vocabulary_size,
                                  embedding_size],
                           name="W_layer1")
nce_weights = tf.Variable(
  tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(
  tf.nn.nce_loss(nce_weights, nce_biases, embed, train_words,
                 num_negs, vocabulary_size))

optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
print("Graph is built...")

if sys_path != "":
    pickled_data_folder = sys_path + "/"
else:
    pickled_data_folder = "CCBOW/"
protocol_2 = True
if protocol_2:
    is_protocol2 = "_Protocol_2"
else:
    is_protocol2 = ""


with open(pickled_data_folder + "Pickled_Train_Data_" + str(vocabulary_size) +
                  "/Train_Data.pickle", "rb") as f:
    train_dat = pickle.load(f)
print("Done pickling train data")

with open(pickled_data_folder + "Pickled_Train_Data_" + str(vocabulary_size) +
                  "/Num_to_Word.pickle", "rb") as f:
    num_to_word = pickle.load(f)
print("Done pickling num_to_wrd")

with open(pickled_data_folder + "Pickled_Train_Data_" + str(vocabulary_size) +
                  "/Word_to_Num.pickle", "rb") as f:
    word_to_num = pickle.load(f)
print("Done pickling word_to_num")

with open(pickled_data_folder + "Pickled_Train_Data_" + str(vocabulary_size) +
                  "/POS_to_num.pickle", "rb") as f:
    pos_to_num = pickle.load(f)
print("Done pickling pos_to_num")

with open(pickled_data_folder + "Pickled_Train_Data_" + str(vocabulary_size) +
                  "/Num_to_POS.pickle", "rb") as f:
    num_to_pos = pickle.load(f)
print("Done pickling num_to_pos")

with open(pickled_data_folder + "Pickled_Train_Data_" + str(vocabulary_size) +
                  "/Counts.pickle", "rb") as f:
    counts = pickle.load(f)

with open(pickled_data_folder + "Pickled_Train_Data_" + str(vocabulary_size) +
                  "/Frequencies.pickle", "rb") as f:
    frequencies = pickle.load(f)

# Hold on to an original version of the training data when we need to reload
# After an epoch
start = datetime.now()
# training_data = copy.deepcopy(train_dat)
print("Time to deepcopy = ", datetime.now() - start)
print("Starting shuffling...")
start = datetime.now()
random.shuffle(train_dat)
print("Time to shuffle = ", datetime.now() - start)
training_data = deque(train_dat)
training_set_size = len(training_data)
print(training_set_size)

saver = tf.train.Saver()

with tf.Session() as sess:
    print("Inside session.")
    sess.run(tf.initialize_all_variables())
    print("Variables initialized...")
    epoch = 0
    num_batches = 1
    saver.restore(
        sess=sess,
        save_path="CCBOW/checkpoints/embeddingSize-200_contextWindow-5_"
                  "numNegs-2_vocabSize-20000_numBatches-24000.ckpt-24000")
    while 300000000 / (num_batches * batch_size) > 1:
        print(num_batches)
        start = datetime.now()
        batch = generate_batch(batch_size, context_window)
        print("Time to generate a batch: ", datetime.now() - start)
        if num_batches % 500 == 0:
            print("Positives loss: ",
                  loss.eval(feed_dict={
                      train_inputs: [t[1] for t in batch],
                      train_words: [t[0] for t in batch]}))
        start = datetime.now()
        optimizer.run(feed_dict={
            train_inputs: [t[1] for t in batch],
            train_words: [t[0] for t in batch]})

        if num_batches % 2000 == 0:
            checkpoint_file = pickled_data_folder + "checkpoints/" + \
                              "embeddingSize-" + str(embedding_size) + \
                              "_contextWindow-" + str(context_window) + \
                              "_numNegs-" + str(num_negs) + \
                              "_vocabSize-" + str(vocabulary_size) + \
                              "_numBatches-" + str(num_batches) + ".ckpt"
            saver.save(sess, checkpoint_file, global_step=num_batches)

        print("Time for batch training: ", datetime.now() - start)

        num_batches += 1






