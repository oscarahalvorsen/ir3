import string
import codecs
import re
from gensim import corpora
from nltk.stem.porter import PorterStemmer

def getText(txtFile):
    f = codecs.open(txtFile, "r", "utf-8")
    return ''.join(f.readlines())

def getDocumentCollection(txtFile):
    dc = re.split('\r\n\s{0,}\r\n', getText(txtFile))
    return [d.translate(str.maketrans('','',string.punctuation+"\n\r\t")).split() for d in dc if (not bool(re.search('gutenberg', d.lower())))]

def stemWords(dc):
    stemmer = PorterStemmer()
    return [[stemmer.stem(word.lower()) for word in d] for d in dc]

def filterOutStopwords(dc, stopWordFile):
    f = codecs.open(stopWordFile, "r", "utf-8")
    stopWords = re.split(',', f.readline())
    dc = [[word for word in d if word not in stopWords] for d in dc]
    return [d for d in dc if d!=[]]

def makeBagOfWords(dc):
    dictionary = corpora.Dictionary()
    return [dictionary.doc2bow(d,allow_update=True) for d in dc]

txtFile = "pg3300.txt"
stopWordFile ="stopwords.txt"
dc = getDocumentCollection(txtFile)
dc = filterOutStopwords(dc, stopWordFile)
print(dc)
print(len(dc))
dc = makeBagOfWords(dc)
print(dc)
print(len(dc))
