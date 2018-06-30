# -*- coding: utf-8 -*-
"""
@CREATETIME: 23/06/2018 14:41 
@AUTHOR: Chans
@VERSION: 
"""
'''
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = ["I come to China to travel",
    "This is a car polupar in China",
    "I love tea and Apple ",
    "The work is to write some papers in science"]
print(vectorizer.fit_transform(corpus))

print(vectorizer.fit_transform(corpus).toarray())
print(vectorizer.get_feature_names())
'''


import  jieba
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words_path = '/Users/Apple/datadata/corpus/stop_words.txt'
with open(stop_words_path,'r',encoding='gbk') as f:
    stpwrdlst = f.read()

stpwrdlst2 = stpwrdlst.splitlines()




str = '沙瑞金赞叹易学习的胸怀，是金山的百姓有福，可是这件事对李达康的触动很大。易学习又回忆起他们三人分开的前一晚，大家一起喝酒话别，易学习被降职到道口县当县长，王大路下海经商，李达康连连赔礼道歉，觉得对不起大家，他最对不起的是王大路，就和易学习一起给王大路凑了5万块钱，王大路自己东挪西撮了5万块，开始下海经商。没想到后来王大路竟然做得风生水起。沙瑞金觉得他们三人，在困难时期还能以沫相助，很不容易。'
str2 = '沙瑞金向毛娅打听他们家在京州的别墅，毛娅笑着说，王大路事业有成之后，要给欧阳菁和她公司的股权，她们没有要，王大路就在京州帝豪园买了三套别墅，可是李达康和易学习都不要，这些房子都在王大路的名下，欧阳菁好像去住过，毛娅不想去，她觉得房子太大很浪费，自己家住得就很踏实。'
jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)

jieba_cut = jieba.cut(str,cut_all=False)
# contents = [ele for ele in jieba_cut]
jieba_cut2 = jieba.cut(str2,cut_all=False)
# contents2 = [ele for ele in jieba_cut2]
# print(jieba_cut)
contents = ''
contents2 = ''
for i in jieba_cut:
    contents += i + ' '
for i in jieba_cut2:
    contents2 += i + ' '

corpus = [contents,contents2]
vector = TfidfVectorizer(stop_words=stpwrdlst2)
tfidf = vector.fit_transform(corpus)
print(tfidf)

wordlist = vector.get_feature_names()#获取词袋模型中的所有词
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = tfidf.toarray()
#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weightlist)):
    print("-------第",i,"段文本的词语tf-idf权重------"  )
    for j in range(len(wordlist)):
        print(wordlist[j],weightlist[i][j])