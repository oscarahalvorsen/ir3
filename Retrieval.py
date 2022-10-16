import string
import codecs
import re
from gensim import corpora, models, similarities
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
dictionary = corpora.Dictionary()

def getText(txtFile):
    f = codecs.open(txtFile, "r", "utf-8")
    return ''.join(f.readlines())

def getDocumentCollection(dc):
    dc = re.split('\r\n\s{0,}\r\n', getText(txtFile))
    return [d.translate(str.maketrans('','',string.punctuation+"\n\r\t")).split() for d in dc if (not bool(re.search('gutenberg', d.lower())))]

def stemWords(dc):
    return [[stemmer.stem(word.lower()) for word in d] for d in dc]

def filterOutStopwords(dc, stopWordFile):
    f = codecs.open(stopWordFile, "r", "utf-8")
    stopWords = re.split(',', f.readline())
    dc = [[word for word in d if word not in stopWords] for d in dc]
    return [d for d in dc if d!=[]]

def makeBagOfWords(dc):
    return [dictionary.doc2bow(d,allow_update=True) for d in dc]


def preprossesText(txtFile, stopWordsFile):
    dc = getDocumentCollection(txtFile)
    dc = stemWords(dc)
    dc = filterOutStopwords(dc, stopWordsFile)
    return makeBagOfWords(dc)

def preprossesQuery(query, stopWordsFile, ):
    q = re.split(' ', query)
    q = [stemmer.stem(word.lower()) for word in q]
    f = codecs.open(stopWordsFile, "r", "utf-8")    
    stopWords = re.split(',', f.readline())
    q = [word for word in q if word not in stopWords]
    return [dictionary.doc2bow(q)]

def getTFIDF(textBow, queryBow):
    tfidf_model = models.TfidfModel(textBow)
    return tfidf_model[queryBow[0]]

txtFile = "pg3300.txt"
stopWordsFile ="stopwords.txt"
textBow = preprossesText(txtFile, stopWordsFile)
print(textBow)

q1 ="How taxes influence Economics"
q1Bow = preprossesQuery(q1, stopWordsFile)
print(q1)
print(getTFIDF(textBow, q1Bow))


