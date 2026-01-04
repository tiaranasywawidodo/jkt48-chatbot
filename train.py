import json
import numpy as np
import nltk
import pickle
import random
import ssl
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Bypass SSL untuk download NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
stemmer = StemmerFactory().create_stemmer()

# 1. Load Data
with open('intents.json') as file:
    data = json.load(file)

words, classes, documents = [], [], []
ignore_words = ['?', '!', '.', ',']

print("Sedang memproses teks...")
for intent in data['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern.lower())
        w = [stemmer.stem(i) for i in w if i not in ignore_words]
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# 2. Persiapan Data Training
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# 3. Membangun Model Deep Learning
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("Memulai Training Model...")
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print("Model berhasil dibuat dan disimpan!")