# encoding=utf-8
import  jieba
import  jieba.posseg as psg
import os
import  re
import csv


corpora_path = '/Volumes/d/data/corpora_data_test/'
output_path = '/Volumes/d/data/jieba_cut'
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




for fs in document_lists:
    document_path = corpora_path+fs
    print(document_path)
    try:
        with open(document_path, 'r', encoding='gb18030') as f:
            results = []
            txt_output = open(output_path + fs, 'w', encoding='utf-8')
            # txt_output2 = open(output_path+"/test/" + fs, 'w', encoding='utf-8')
            for i,line in enumerate(f):
                r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
                line = re.sub(r1,'',line)
                if len(line)>1:
                    keywords_cut1 = jieba.cut(line)
                else:
                    continue
                # keywords_cut2= psg.cut(line)
                # results = results.join(keywords_cut)
                results1 = " ".join(keywords_cut1)
                # results2 = " ".join(keywords_cut1)
                # print("------")
                qq1 =map(deal_special,results1)
                # qq1 = results1
                # print("--------")
                x1 = str(list(qq1))
                # x2 = str(list(qq2))
                x1 = x1.replace(r"\n","").replace("'","").replace(",","").replace('\]',"").replace('\[',"").strip()
                # x2 = x2.replace(r"\n", "").replace("'", "").replace(",", "").replace(']', "").replace('[', "").strip()
                # print("第"+str(i)+"个"+str(x) )
                txt_output.write(x1+'\n')
                # txt_output2.write(x2 + '\n')
            txt_output.close()
            # txt_output2.close()
        # with open(output_path+fs,'a',encoding='utf-8') as ff:
        #     for row in range(len(x)) :
        #         ff.write(x[row]+'\n')

            # unique_keywords = codecs.open(new_txt_path.decode('utf-8'), 'w', encoding='utf-8')
    except Exception  as e:
          print("Exception: {}".format(e))



