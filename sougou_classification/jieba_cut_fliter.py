# coding=utf-8
"""
调用jieba分词，完成搜索文本的分词。同时只保留n,v,j三种词性。
注意输入的文件为纯文本格式，最好一个用户的搜索历史为一行。
注意路径
"""
import pandas as pd
import jieba.analyse
import time
import jieba
import jieba.posseg
import os, sys
def input(trainname):
    traindata = []
    with open(trainname, 'r',encoding='gb18030') as f:
        line = f.readline()
        count = 0
        while line:
            try:
                traindata.append(line)
                count += 1
            except:
                print("error:", line, count)
            line=f.readline()
    return traindata
start = time.clock()
root_path = '/Volumes/d/data/sougoudata_ori/'
filepath = root_path+'test_querylist.csv'
QueryList = input(filepath)

writepath =root_path+ 'testfile.csv'
csvfile = open(writepath, 'w',encoding='gb18030')
#parallel:speed up
jieba.enable_parallel(processnum=4)
POS = {}
for i in range(len(QueryList)):
    s = []
    str = ""
    words = jieba.posseg.cut(QueryList[i])# 带有词性的精确分词模式
    allowPOS = ['n','v','j']
    for word, flag in words:
        POS[flag]=POS.get(flag,0)+1
        if (flag[0] in allowPOS) and len(word)>=2:
            str += word + " "
    s.append(str)
    csvfile.write(" ".join(s)+'\n')
csvfile.close()
print(POS)

end = time.clock()
print("total time: %f s" % (end - start))


# seg_list = jieba.cut("陶喆下载", cut_all=False)
# print("Default Mode: " + "/ ".join(seg_list))  # 默认模式
#
# words = jieba.posseg.cut("陶喆下载")
# for word, flag in words:
#     print('%s %s' % (word, flag))