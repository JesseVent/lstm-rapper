import numpy as np
from keras.models import model_from_yaml
from random import randint
import random
import sys

with open("sonnets.txt") as corpus_file:
    corpus = corpus_file.read()
print("Loaded a corpus of {0} characters".format(len(corpus)))

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
# Get a unique identifier for each char in the corpus, then make some dicts to ease encoding and decoding
chars = sorted(list(set(corpus)))
encoding = {c: i for i, c in enumerate(chars)}
decoding = {i: c for i, c in enumerate(chars)}

# Some variables we'll need later
num_chars = len(chars)
sentence_length = 50
corpus_length = len(corpus)

with open("model.yaml") as model_file:
    architecture = model_file.read()

model = model_from_yaml(architecture)
model.load_weights("weights-00-1.354.hdf5")
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

start_index = random.randint(0, len(corpus) - sentence_length - 1)
diversity = 0.5
generated = ''
sentence = corpus[start_index: start_index + sentence_length]
generated += sentence
print('----- Generating with seed: "' + sentence + '"')
sys.stdout.write(generated)

for i in range(1200):
    x_pred = np.zeros((1, sentence_length, num_chars))
    for t, char in enumerate(sentence):
        x_pred[0, t, encoding[char]] = 1.
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = decoding[next_index]
    generated += next_char
    sentence = sentence[1:] + next_char
    sys.stdout.write(next_char)
    sys.stdout.flush()
print()