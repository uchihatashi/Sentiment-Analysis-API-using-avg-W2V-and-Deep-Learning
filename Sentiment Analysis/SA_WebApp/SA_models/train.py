# importing libraries
from text_process import text_preprocessing # own class
from text_process import print_10

import pandas as pd 
import pickle
import numpy as np
import re
from tqdm import tqdm

import nltk
from nltk import WordNetLemmatizer
import gensim
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt
#%matplotlib inline




# loading data 
df = pd.read_csv("airline_sentiment_analysis.csv", encoding="latin-1")
X = df["text"]
y = df["airline_sentiment"].map({"positive":1, "negative":0})


# text Preprocessing 
preprocessed_X = []
for text in tqdm(X.values):
    preprocessed_X.append( text_preprocessing(text) )
#print_10(preprocessed_X)

#making it into list of words
list_of_words=[]
for words in tqdm(preprocessed_X):
    list_of_words.append(words.split())
    
""" Creating and initilizing word2vec 
min_count = 5 considers only words that occured atleast 5 times and size=50 as we have small dataset 
"""
w2v_model = Word2Vec(size=50, min_count=5, workers=4)
w2v_model.build_vocab(list_of_words)

total_corpus = w2v_model.corpus_count
#print(total_corpus)

# traning word2vec
w2v_model.train(list_of_words, total_examples=total_corpus, epochs=w2v_model.iter) 

# Save the w2v model
w2v_model.save('w2vmodel')
# Load the w2v model
new_w2v_model = gensim.models.Word2Vec.load('w2vmodel')

"""print(new_w2v_model["great"].shape)
print('='*100,"\n")
print(new_w2v_model["great"])
print('='*100, "\n")
print(new_w2v_model.most_similar("great"))
print('='*100, "\n")"""

def createAvgWordVector(each_text, size=50):
    """
    """
    vec = np.zeros(size).reshape(size)
    count = 0.
    for word in each_text:
        try:
            vec += new_w2v_model[word].reshape(size)
            count += 1.
            
        except KeyError: # handling the case where the each_text is not
                         # in the corpus. useful for testing.
            continue
    if count != 0: 
        vec /= count
        
    return vec

#Splitting for training and testing
x_train, x_test, y_train, y_test = train_test_split(np.array(list_of_words), np.array(y), test_size=0.3)

x_train_v2w = list(map(createAvgWordVector, x_train))
x_test_v2w = list(map(createAvgWordVector, x_test))

#converting into ndarray
x_train_v2w = np.array(x_train_v2w)
x_test_v2w = np.array(x_test_v2w)

# to prevent errors in further process in traing and testing for the model
assert(x_train_v2w.shape[0] == y_train.shape[0]), "The number of train_text is not equal to the number of labels."
assert(x_train_v2w.shape[1] == (50)), "The dimensions of the train_text(w2v) are not 50"
assert(x_test_v2w.shape[0] == y_test.shape[0]), "The number of test text is not equal to the number of labels."
assert(x_test_v2w.shape[1] == (50)), "The dimensions of the test text(w2v) are not 50"

def model1():
    """
    """
    model = Sequential()
    model.add(Dense(40, activation='relu', input_dim=50))
    model.add(Dropout(0.4))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(Adam(lr = 0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

#initilizing the model1
model1 = model1()

#fitting the model
history = model1.fit(x_train_v2w, y_train, validation_split = 0.2, epochs=100, batch_size=64, verbose=2, shuffle = 1)

# to plot the train loss and valdation loss for model1 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','test'])
plt.title('MODEL1: Loss')
plt.xlabel('epoch')
plt.show()

# to plot the train accuracy and valdation accuracy for model1 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','test'])
plt.title('MODEL1: Accuracy')
plt.xlabel('epoch')
plt.show()

# evaluating the model1 by using test data
score = model1.evaluate(x_test_v2w, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#saving model
model1.save("model1")

"""# OPTIONAL MODEL2:

def model2():
    model = Sequential()
    model.add(Dense(40, activation='relu', input_dim=50))
    model.add(Dropout(0.4))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(RMSprop(lr = 0.01), loss='mse', metrics=['accuracy'])
    print(model.summary())
    return model

model2 = model2()

history2 = model2.fit(x_train_v2w, y_train, validation_split = 0.2, epochs=100, batch_size=64, verbose=2, shuffle = 1)

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.legend(['training','test'])
plt.title('MODEL2: Loss')
plt.xlabel('epoch')

plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.legend(['training','test'])
plt.title('MODEL2: Accuracy')
plt.xlabel('epoch')

score2 = model2.evaluate(x_test_v2w, y_test, verbose=1)

print('Test score:', score2[0])
print('Test accuracy:', score2[1])

#to save model
model2.save("model2")

#to load model https://www.tensorflow.org/guide/keras/save_and_serialize
reconstructed_model = keras.models.load_model("model2")

score2 = reconstructed_model.evaluate(x_test_v2w, y_test, verbose=1)

result = reconstructed_model.predict_classes(x_test_v2w[0:1, :])
print(result)
print(y_test[9])
"""


print("\n"*2, "*"*10, "ALL DONE", "*"*10, "\n"*2,)