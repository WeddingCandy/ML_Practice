# encoding=utf-8
import  jieba
import os
import  re


corpora_path = '/Volumes/d/data/corpora_data_test/'
output_path = '/Volumes/d/data/jieba_cut'
document_lists =  os.listdir(corpora_path)

pattern = re.compile("[^\[\],]")

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
            for i,line in enumerate(f):
                # print("line的内容："+ line)
                keywords_cut = jieba.cut(line,cut_all=False,HMM=True)
                # results = results.join(keywords_cut)
                results.append(" ".join(keywords_cut))
                # print("------")
                # qq = map(pattern,results)
                # print("--------")
                # x= str(list(qq))
                qq=  str(results).replace(r"\n","").replace("'","").replace(",","").replace(']',"").replace('[',"").strip()
                # print("x"+x)
                # print("第"+str(i)+"个"+qq )
        with open(output_path+fs,'w',encoding='utf-8') as ff:
            output = " ".join(qq)
            # print("output    "+output)
            ff.write(output)

            # unique_keywords = codecs.open(new_txt_path.decode('utf-8'), 'w', encoding='utf-8')
    except Exception  as e:
          print("Exception: {}".format(e))




