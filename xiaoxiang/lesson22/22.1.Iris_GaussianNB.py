#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures#StandardScaler:标准化
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_20newsgroups


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    data_type = 'news'  # iris,news,car

    if data_type == 'car':
        colmun_names = 'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability'#元祖的写法
        data = pd.read_csv('/Users/Apple/PycharmProjects/learn_ml/lesson22/car.data', header=None, names=colmun_names)
        for col in colmun_names:
            data[col] = pd.Categorical(data[col]).codes #Categorical 能把所有的种类找出来；.codes能进行编码
        x = data[list(colmun_names[:-1])] #取列时，实际上是左闭右开
        y = data[colmun_names[-1]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        model = MultinomialNB(alpha=1) #MultinomialNB的参数
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        print('CAR训练集准确率：', accuracy_score(y_train, y_train_pred)) #测试分类器的分类正确率
        y_test_pred = model.predict(x_test)
        print('CAR测试集准确率：', accuracy_score(y_test, y_test_pred))

    if data_type == 'news':
        news = fetch_20newsgroups(subset='all')
        x_train, x_test, y_train, y_test = train_test_split(news.data,news.target,test_size= 0.25,random_state=33)
        vec = CountVectorizer()
        X_train = vec.fit_transform(x_train)
        X_test = vec.transform(x_test)
        model = MultinomialNB(alpha=1)
        model.fit(X_train,y_train)
        y_predict = model.predict(X_test)
        print('news训练集准确率：', accuracy_score(y_train, y_predict)) #测试分类器的分类正确率
        y_test_pred = model.predict(x_test)
        print('news测试集准确率：', accuracy_score(y_test, y_test_pred))




    else:
        feature_names = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度','类型'
        data = pd.read_csv('/Users/Apple/PycharmProjects/learn_ml/lesson9_regression/iris.data', header=None, names=feature_names)
        x, y = data[list(feature_names[:-1])], data[feature_names[-1]] #xy合在一起写
        y = pd.Categorical(values=data['类型']).codes
        features = ['花萼长度', '花萼宽度']
        x = x[features]
        x, x_test, y, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        priors = np.array((1,2,4), dtype=float)
        priors /= priors.sum()
        gnb = Pipeline([
            ('sc', StandardScaler()),
            ('poly', PolynomialFeatures(degree=1)),
            ('clf', GaussianNB(priors=priors))])    # 由于鸢尾花数据是样本均衡的，其实不需要设置先验值
        # gnb = KNeighborsClassifier(n_neighbors=3).fit(x, y.ravel())
        gnb.fit(x, y.ravel()) # ravel（）将y列变成一列
        y_hat = gnb.predict(x)
        print('IRIS训练集准确度: %.2f' % (100 * accuracy_score(y, y_hat))) #后面两个百分号是取百分数的意思
        y_test_hat = gnb.predict(x_test)
        print('IRIS测试集准确度：%.2f%%' % (100 * accuracy_score(y_test, y_test_hat)))  # 画图

        N, M = 500, 500     # 横纵各采样多少个值
        x1_min, x2_min = x.min() #x是两维的
        x1_max, x2_max = x.max()
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
        x_grid = np.stack((x1.flat, x2.flat), axis=1)   # 测试点

        mpl.rcParams['font.sans-serif'] = ['simHei']
        mpl.rcParams['axes.unicode_minus'] = False
        cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        y_grid_hat = gnb.predict(x_grid)                  # 预测值
        y_grid_hat = y_grid_hat.reshape(x1.shape)
        plt.figure(facecolor='w')
        plt.pcolormesh(x1, x2, y_grid_hat, cmap=cm_light)     # 预测值的显示
        plt.scatter(x[features[0]], x[features[1]], c=y, edgecolors='k', s=30, cmap=cm_dark)
        plt.scatter(x_test[features[0]], x_test[features[1]], c=y_test, marker='^', edgecolors='k', s=40, cmap=cm_dark)

        plt.xlabel(features[0], fontsize=13)
        plt.ylabel(features[1], fontsize=13)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title('GaussianNB对鸢尾花数据的分类结果', fontsize=18)
        plt.grid(True, ls=':', color='#202020')
        plt.show()
