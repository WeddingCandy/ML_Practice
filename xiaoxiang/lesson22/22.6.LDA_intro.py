# !/usr/bin/python
# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities
from pprint import pprint
import warnings

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    # try:
    #     f = open('/Users/Apple/PycharmProjects/learn_ml/lesson22/LDA_test.txt','r')
    # finally:
    #     if f:
    #         f.close()
    f = open('/Users/Apple/PycharmProjects/learn_ml/xiaoxiang/lesson22/LDA_test.txt', 'r',encoding='utf-8')
    stop_list = set('for a of the and to in'.split())
    texts = [line.strip().split() for line in f]
    print('Before')
    pprint(texts)
    print('After')
    texts = [[word for word in line.strip().lower().split() if word not in stop_list] for line in f]
    '''
    strip() 只能去除头尾
    '''
    #等价
    # text = []
    # for line in f:
    #     for word in line.strip().lower().split():
    #         if word not in stop_list:
    #             text.append(word)
    print('Text = ')
    pprint(texts)

    '''
        corpora:创建字典
        corpora.Dictionary() 主要将一个array对象转换成字典
        corpora.add_documents() 补充新的文档到字典中
        doc2bow() 获得每一篇文档对应的稀疏向量,变成词袋，分别为id和出现次数
        a = models.TfidfModel(corpus) 使用tf-idf模型得出该评论集的tf-idf模型
        b = a[corpus]  此处已经计算得出所有评论的tf-idf 值
    '''
    dictionary = corpora.Dictionary(texts)
    print(dictionary)
    V = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpus_tfidf = models.TfidfModel(corpus)[corpus]

    print('TF-IDF:')
    for c in corpus_tfidf:
        print(c)
    '''
    models.LsiModel()   文档相似度
    print_topics()  输出一共有的主题,设定主题词后会得出主要的词以及权重系数
    similarities.MatrixSimilarity()
    '''

    print('\nLSI Model:')
    lsi = models.LsiModel(corpus_tfidf, num_topics=3, id2word=dictionary)
    topic_result = [a for a in lsi[corpus_tfidf]]
    pprint(topic_result)
    print('LSI Topics:')
    pprint(lsi.print_topics(num_topics=3, num_words=5))
    similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])   # similarities.Similarity()
    print('Similarity:')
    pprint(list(similarity))
    '''
    指定主题数量 num_topics
    models.LdaModel()
    '''
    print('\nLDA Model:')
    num_topics = 10
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha='auto', eta='auto', minimum_probability=0.001, passes=10)
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    print('Document-Topic:\n')
    pprint(doc_topic)
    for doc_topic in lda.get_document_topics(corpus_tfidf):
        print(doc_topic)
    for topic_id in range(num_topics):
        print('Topic', topic_id)
        pprint(lda.get_topic_terms(topicid=topic_id))#词的编号
        pprint(lda.show_topic(topic_id))#得出整个主题的关键词
    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    print('Similarity:')
    pprint(list(similarity))
    '''
    models.HdpModel()  
    hda.print_topics() 可以输出主题以及主题的
    '''
    hda = models.HdpModel(corpus_tfidf, id2word=dictionary)
    topic_result = [a for a in hda[corpus_tfidf]]
    print('\n\nUSE WITH CARE--\nHDA Model:')
    pprint(topic_result)
    print('HDA Topics:')
    print(hda.print_topics(num_topics=2, num_words=5))
