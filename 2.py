import jieba.posseg as pseg 
import codecs
from gensim import corpora,models,similarities

doc_stop='D:/stopword.txt'
stopwords = open(doc_stop,encoding='utf-8').read()
stopwords = stopwords.split()
stopwords
