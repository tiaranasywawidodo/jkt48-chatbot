import nltk
# Tambahkan dua baris ini:
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('wordnet')
import pickle
import numpy as np
import json
from tensorflow.keras.models import load_model
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Memuat model dan data hasil training
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
stemmer = StemmerFactory().create_stemmer()

def predict_class(sentence):
    # Preprocessing input user
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [stemmer.stem(word) for word in sentence_words]
    
    # Feature Extraction (Bag of Words)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: bag[i] = 1
    
    # Prediksi
    res = model.predict(np.array([bag]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]