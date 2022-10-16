import random
import string
import codecs
import re

def getText(txtFile):
    f = codecs.open(txtFile, "r", "utf-8")
    return ''.join(f.readlines())

def getDocumentCollection(txtFile):
    dc = re.split('\r\n\s{0,}\r\n', getText(txtFile))
    prossesedDc = []
    for d in dc:
        d = d.translate(str.maketrans('','',string.punctuation+"\n\r\t"))
        if not bool(re.search('gutenberg', d.lower())):
            prossesedDc.append(d.split())
    return prossesedDc

txtFile = "pg3300.txt"
dc = getDocumentCollection(txtFile)
print(dc)
print(len(dc))

