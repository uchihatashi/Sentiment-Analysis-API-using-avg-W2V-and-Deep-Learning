import re
from nltk import WordNetLemmatizer
import numpy as np

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
lemmatizer = WordNetLemmatizer()


def decontracted(phrase):
    """
    """
    
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    return phrase


def text_preprocessing(text):
    """
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text) # removing any links in the text
    text = re.sub(r"@\S+", "", text) # we need to remove all username as it dosen't contribute to SA 
    text = text.replace("#", "") # removing hash but keeping text #fabulous => fabulous    
    text = decontracted(text) 
    text = re.sub("\S*\d\S*", "", text).strip() # remove words with numbers python: https://stackoverflow.com/a/18082370/4084039
    text = re.sub('[^A-Za-z0-9]+', ' ', text) # remove spacial character: https://stackoverflow.com/a/5843547/4084039
    text = ' '.join(e for e in text.split() if e not in stopwords) # using own created stopwords 
    # removing any words which have less than 3 character and lemmatizer those words having higher than 2 character
    text = ' '.join(lemmatizer.lemmatize(e) for e in text.split() if len(e) > 2) 
        
    return text


def print_10(lists):
    for i in range(10):
        print(lists[i], "\n")

def createAvgWordVector(each_text, new_w2v_model, size=50):
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