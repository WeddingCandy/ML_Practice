# -*- coding:UTF-8 -*
import  requests,sys
from bs4 import  BeautifulSoup
import pandas as pd
import encodings
import re
import pprint

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36'}

excel_sheet = pd.read_excel('C:/Users/thinkpad/Desktop/labels.xlsx',sheet_name = 'url',header= 0)
# excel_sheet = pd.read_excel(r'/Users/Apple/Desktop/working/0 华院资料/HIVE库表/9 用户浏览标签 -20180319.xlsx',sheet_name = 'url',header= 0)
colums_aim = excel_sheet.iloc[:,3:4]
labels_ori = "http://" + colums_aim
labels = labels_ori.values.tolist()
for id in range(len(labels)):
    try:
        req = requests.get(url=labels[id][0], headers=headers ,timeout=2)
        print('have read'+str(id))
    except Exception  as e:
        print("Exception: {}".format(e))
        print('第%d个label无法爬取' %(str(id)))
        continue
    else:
        data =req.json


    # html0 =req.content.decode('gbk', 'ignore')
    html = req.text.encode(req.encoding).decode('utf-8')
    bf = BeautifulSoup(html, "html.parser")
    meta_all = bf.find_all('meta')
    meta_tostring = str(meta_all)
    # string = '<meta content="保险，平安保险，车险，贷款，理财，信用卡，意外保险，重疾险，小额贷款，信用贷款，投资理财，个人理财，汽车保险，商业保险，少儿保险，健康保险，旅游保险，人寿保险, 医疗保险，平安普惠，平安信用卡，平安车险，平安银行" name="keywords">'
    meta_keyword = re.finditer(
        re.compile(r'([\u4e00-\u9fa5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]+){1,}'),
        meta_tostring)
    keep = []
    for i in meta_keyword:
        print(i.group() + '\n')
        keep.append(i.group())
    keep = list(set(keep))
    print('第%s个label已完成' %(str(id)))