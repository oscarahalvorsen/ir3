import string
import codecs
import re
import random
import nltk
from gensim import corpora, models, similarities
from nltk.stem.porter import PorterStemmer
import numpy as np
from itertools import chain

stemmer = PorterStemmer()
dictionary = corpora.Dictionary()

txtFile = "pg3300.txt"
stopWordsFile ="stopwords.txt"
stopWords = codecs.open(stopWordsFile, "r", "utf-8").read()

# Task 1
random.seed(123) #1.0, fix random number generator
raw = codecs.open(txtFile, "r", "utf-8").read() #1.1, open and load the file (it’s UTF-8 encoded) using codecs
raw_tokenized = nltk.BlanklineTokenizer().tokenize(raw) # 1.2 partition file into separate paragraphs
dc_raw = [] #copy of the original paragraphs for displaying querying results
dc = []
for d in raw_tokenized:
    if (not bool(re.search('gutenberg', d.lower()))): # 1.3 remove all paragraphs containing the word “Gutenberg” 
        dc_raw.append(d)
        d=d.translate(str.maketrans('','',string.punctuation+"\n\r\t")).split() # 1.4 remove text punctuation
        dc.append([stemmer.stem(word.lower()) for word in d]) # 1.5 stems the words using PorterStemmer
freqDist = nltk.FreqDist(list(chain.from_iterable(dc))) # 1.6 create class #freqDist that contains all words and their respective count in dc
print(freqDist["tax"])

# Task 2
# 2.1 remove stopwords and add to a dictionary
dc = [[word for word in d if word not in stopWords] for d in dc]
dic = corpora.Dictionary(dc)
print(dc)

bow = [dictionary.doc2bow(d,allow_update=True) for d in dc] # 2.2 make Bag-of-Words
print(bow)

#Task 3
#TF-IDF model
tfidf_model = models.TfidfModel(bow) # 3.1 setup tf-idf model
tfidf_corpus = tfidf_model[bow] # 3.2 map bow
matSim = similarities.MatrixSimilarity(tfidf_corpus, dic) # 3.3 construct matrixSimilarity

#LSI model
# 3.4 same as 3.1 to 3.3 but for LSI
lsi_model= models.LsiModel(tfidf_corpus, id2word=dic,num_topics=100) # 3.1 setup lsi model
lsi_corpus = lsi_model[bow] # 3.2 map bow
lsi_MatSim= similarities.MatrixSimilarity(lsi_corpus) # 3.3 construct matrixSimilarity

print(lsi_model.show_topics(3)) # Task 3.5

# Task 4
# 4.1 apply all necessary transformations: remove punctuations, tokenize, stem and convert to BOW
query = "What is the function of money?" 
q = re.split(' ', query)
q = [stemmer.stem(word.lower()) for word in q]
q = [word for word in q if word not in stopWords]
qBow= dictionary.doc2bow(q)

q_tfidf = tfidf_model[qBow] # 4.2 convert BOW to TF-IDF representation

# 4.3 Report top 3 the most relevant paragraphs for the query "What is the function of money?" using tf-idf model
tfidf_index = similarities.MatrixSimilarity(tfidf_corpus)
doc2similarity = enumerate(tfidf_index[q_tfidf])
q_sorted = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]  

def truncate(text):
    return ' '.join(re.split(r'(?<=[.:;])\s', text)[:5])

print("\nTF-IDF model")
# this returned 682, 817 and 683
for i in range(3):
    print(f"\n paragraph [{q_sorted[i][0]}] \n {truncate(dc_raw[q_sorted[re.I][0]])} ")

# 4.4 Report top 3 the most relevant paragraphs for the query "What is the function of money?" using lsi model
lsi_query = lsi_model[qBow]
topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3] 
doc3similarity = enumerate(lsi_MatSim[lsi_query])
lsi_p = sorted(doc3similarity, key=lambda kv: -kv[1])[:3]

# the 3 most relevant topics
print("\nLSI model")
for i in range(3):
    print( lsi_model.show_topics()[topics[i][0]])
    print(f"\n paragraph [{lsi_p[i][0]}] \n {truncate(dc_raw[lsi_p[i][0]])} ")

