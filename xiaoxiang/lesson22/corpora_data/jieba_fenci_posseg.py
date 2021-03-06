# encoding=utf-8
import  jieba
import  jieba.posseg as psg
import os
import  re
import csv



document_path = '/Volumes/d/data/corpora_data_test/news.allsites.1680806.txt'
# corpora_path = '/Volumes/d/data/corpora_data/'
corpora_path = '/Volumes/d/data/corpora_data_test/'
output_path = '/Volumes/d/data/jieba_cut_test/'
document_lists =  os.listdir(corpora_path)
def deal_special(s):
    pattern = re.compile(r"[^\u4e00-\u9f5a]")
    return pattern.sub('',s)

def deal_others(s):
    pattern = re.compile("[\[\],']")
    return pattern.sub('',s)

def len_big_zero(item):
    if len(item)>0:
       return 1
    return 0

# jieba.load_userdict('/Volumes/d/data/stopwords_for_jieba.txt')


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    sentence_seged = sentence
    stopwords = stopwordslist('/Volumes/d/data/stopwords_for_jieba.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    r3 = '\s{2,}'
    outstr = re.sub(r3, '', outstr)
    return outstr



for fs in document_lists:
    document_path = corpora_path+fs
    print(document_path)
    try:
        with open(document_path, 'r', encoding='gb18030') as f:
            results = []
            POS ={}
            txt_output = open(output_path + fs, 'w', encoding='utf-8')
            # txt_output2 = open(output_path+"/test/" + fs, 'w', encoding='utf-8')
            for i,line in enumerate(f):
                r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
                line = re.sub(r1,'',line)
                r2 = re.compile(r"[^\u4e00-\u9f5a]")
                line = r2.sub(' ',line)

                if len(line)>1:
                    keywords_cut1 = jieba.posseg.cut(line)
                    allowPOS = ['n', 'v', 'j']
                    strx = ""
                    for w in keywords_cut1:
                        print(w)
                        POS[w.flag] = POS.get(w.flag,0) + 1
                        """
                        classCount.get(voteIlabel,0)返回字典classCount中voteIlabel元素对应的值,若无，则进行初始化
                        """

                        if (w.flag[0] in allowPOS) and len(w.word) >= 2:
                            strx += w.word + " "
                        print(strx)

                    # line_seg = seg_sentence(keywords_cut1)

                else:
                    continue
                # results1 = " ".join(keywords_cut1)
                # qq1 = results1
                # qq1 = line_seg
                # pattern = re.compile(r"[^\u4e00-\u9f5a]")
                # qq1 = pattern.sub(' ',results1)
                # qq1 =map(deal_special,results1)                # print("--------\n"+str(type(qq1)))
                x1 = list(qq1) # 不能随便list，可能会把内部的元素都截取出来单独成为一个元素
                x1 = strx
                # print(qq1)
                # print('33'+ i for i in x1)
                # x1 = str(x1).replace(r"\n","").replace("'","").replace(",","").replace('\]',"").replace('\[',"").strip()
                # print("第"+str(i)+"个"+str(x) )
                txt_output.write(qq1+'\n')
                # txt_output2.write(x2 + '\n')
            txt_output.close()
            # txt_output2.close()
        # with open(output_path+fs,'a',encoding='utf-8') as ff:
        #     for row in range(len(x)) :
        #         ff.write(x[row]+'\n')

            # unique_keywords = codecs.open(new_txt_path.decode('utf-8'), 'w', encoding='utf-8')
    except Exception  as e:
          print("Exception: {}".format(e))