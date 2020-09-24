import jieba.posseg as pseg
import codecs
from gensim import corpora, models, similarities

stop_words = 'D:/stopword.txt'   #停用词表的引入
stopwords = codecs.open(stop_words,'r',encoding='utf-8').readlines()
stopwords = [ w.strip() for w in stopwords ]
stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']  #选择分词
def fenci_5288(filename):   # 定义分词操作函数
    result = []
    with open(filename, 'r',encoding='utf-8') as f:
        text = f.read()
        words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

filenames = [       #读取被比较文章位置
            'orig_0.8_add.txt',
            'orig_0.8_del.txt',
            'orig_0.8_dis_1.txt',
            'orig_0.8_dis_3.txt',
            'orig_0.8_dis_7.txt',
            'orig_0.8_dis_10.txt',
            'orig_0.8_dis_15.txt',
            'orig_0.8_mix.txt',
            'orig_0.8_rep.txt',
            ]
corpus = []
for each in filenames:   #分别将需要校对的文件进行读取
    corpus.append(fenci_5288(each))
print(len(corpus))#分词后的长度测量


dictionary = corpora.Dictionary(corpus)#创建IF-TDF模型
doc_vectors = [dictionary.doc2bow(text) for text in corpus]


tfidf = models.TfidfModel(doc_vectors)
tfidf_vectors = tfidf[doc_vectors]

#
# query = fenci('D:/软件工程/test/orig.txt')
# query_bow = dictionary.doc2bow(query)
# index = similarities.MatrixSimilarity(tfidf_vectors)
# sims = index[query_bow]
# print(list(enumerate(sims)))

lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=2)
lsi.print_topics(2)
lsi_vector = lsi[tfidf_vectors]
query = fenci_5288('orig.txt')
query_bow = dictionary.doc2bow(query)
query_lsi = lsi[query_bow]
index = similarities.MatrixSimilarity(lsi_vector)
sims = index[query_lsi]
print(list(enumerate(sims)))
