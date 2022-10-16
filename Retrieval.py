import string
import codecs
import re
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
    
txtFile = "pg3300.txt"
dc = getDocumentCollection(txtFile)
print(stemWords(dc))
print(len(dc))