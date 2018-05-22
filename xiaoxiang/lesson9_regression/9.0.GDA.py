#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn import metrics #评估方法  AUC,ROC
import math
import numpy as np
if __name__ == "__main__":

    learning_rate = 0.001
    list1 =[]
    list2 = []
    for a in range(1,100):
        list2.append(math.sqrt(a))
        cur = 0
        b=1000
        for i in range(b):
            cur -= learning_rate*(cur**2 -a)
        list1.append(cur)

        # print(list1)
        # print(list2)
        print('%d的平方根(近似)为：%.8f，真实值是：%.8f' % (a,cur,math.sqrt(a)))
    array1 = np.array(list1, dtype=int)
    array2 = np.array(list2, dtype=int)

    # print(len(array1),len(array2))
    c = metrics.accuracy_score(array2,array1)   #只能是整数
    print('正确率在b= %d 的情况下为 %.8f' % ( b,c))
    # learning_rate = 0.01
    # for a in range(1,100):
    #     cur = 0
    #     for i in range(1000):
    #         cur -= learning_rate*(cur**2 - a)
    #     print(' %d的平方根(近似)为：%.8f，真实值是：%.8f' % (a, cur, math.sqrt(a)))
