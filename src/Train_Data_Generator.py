import collections
import re
import os
import pickle
from datetime import datetime


def read_data(filename):
    sentences_and_pos = pickle.load(open(filename, 'rb'))
    sentences = [t[0] for t in sentences_and_pos]
    pos = [t[1] for t in sentences_and_pos]
    return sentences, pos

vocabulary_size = 20000

def build_word_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reverse_dictionary


def build_pos_dataset(pos_sentences):
    pos_to_num = {}
    index = 0
    for sentence in pos_sentences:
        for pos in sentence:
            if pos in pos_to_num:
                continue
            else:
                pos_to_num[pos] = index
                index += 1
    num_to_pos = {}
    for key in pos_to_num.keys():
        num_to_pos[pos_to_num[key]] = key
    return pos_to_num, num_to_pos


all_sentences = []
all_pos = []

which_file_num = 0
in_dir = '../Data/Pickled_Sentences/'
for read_file in os.listdir(in_dir):
    if read_file[0] == ".":
        continue
    if which_file_num == 30:
        break
    tmp_sentences, tmp_pos = read_data(in_dir + read_file)
    all_sentences += tmp_sentences
    all_pos += tmp_pos
    print("Done with file: ", which_file_num)
    which_file_num += 1

print("Num sentences: ", len(all_sentences))
words = [item for sublist in all_sentences for item in sublist]
num_words = len(words)
print(num_words)

cnt, word_to_num, num_to_word = build_word_dataset(words)
pos_to_num, num_to_pos = build_pos_dataset(all_pos)
print(cnt[:10])
print(num_to_word[0])
del words  # Hint to reduce memory.


def build_train_dataset(sentences, pos_sentences, word_dict):
    def translate_word(word):
        if word in word_dict:
            return word_dict[word]
        else:
            return 0

    def convert_sentence_to_numbers(sentence, pos_sentence):
        return ([translate_word(word) for word in sentence],
                [pos_to_num[pos] for pos in pos_sentence])
    return [convert_sentence_to_numbers(s, pos_s)
            for s, pos_s in zip(sentences, pos_sentences)]

start = datetime.now()
count_dict = {}
for c in cnt:
    count_dict[c[0]] = c[1]
print("Time taken to produce count dictionary: ", datetime.now() - start)

start = datetime.now()
freq_dict = {}
for w in count_dict:
    freq_dict[w] = count_dict[w]/num_words
print("Time taken to produce freq dictionary: ", datetime.now() - start)

assert(len(all_sentences) == len(all_pos))
train_dat = build_train_dataset(all_sentences, all_pos, word_to_num)
print("Done translating data to numbers")

out_dir = "Data/Pickled_Train_Data_" + str(vocabulary_size)

with open("../" + out_dir + "/Train_Data.pickle", "wb+") as f:
    pickle.dump(train_dat, f, protocol=2)
print("Done pickling train data")

with open("../" + out_dir + "/Num_to_Word.pickle", "wb+") as f:
    pickle.dump(num_to_word, f, protocol=2)
print("Done pickling num_to_wrd")

with open("../" + out_dir + "/POS_to_num.pickle", "wb+") as f:
    pickle.dump(pos_to_num, f, protocol=2)
print("Done pickling pos_to_num")

with open("../" + out_dir + "/Num_to_POS.pickle", "wb+") as f:
    pickle.dump(num_to_pos, f, protocol=2)
print("Done pickling num_to_pos")

with open("../" + out_dir + "/Word_to_Num.pickle", "wb+") as f:
    pickle.dump(word_to_num, f, protocol=2)
print("Done pickling word_to_num")

with open("../" + out_dir + "/Counts.pickle", "wb+") as f:
    pickle.dump(count_dict, f, protocol=2)

with open("../" + out_dir + "/Frequencies.pickle", "wb+") as f:
    pickle.dump(freq_dict, f, protocol=2)
