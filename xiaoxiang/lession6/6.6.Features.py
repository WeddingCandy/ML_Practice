# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA #PCA降维
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
#SelectKBest:选择排名前几的项 ;SelectPercentile:选择前百分之几的项 ；chi2：处理分类问题 ;f_regression :回归用的指标
from sklearn.linear_model import LogisticRegressionCV #LR交叉验证
from sklearn import metrics #评估方法  AUC,ROC
from sklearn.model_selection import train_test_split #随机划分训练集和测试集
from sklearn.pipeline import Pipeline#将多个学习器组成流水线
from sklearn.preprocessing import PolynomialFeatures #生成多项式
from sklearn.manifold import TSNE #TSNE降维
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def extend(a, b):
    return 1.05*a-0.05*b, 1.05*b-0.05*a


if __name__ == '__main__':
    stype = 'pca'
    # stype = 'tsne'
    # stype = 'SelectKBest'
    print(os.path)
    pd.set_option('display.width', 200)
    data = pd.read_csv('/Users/Apple/PycharmProjects/learn_ml/lession6/iris.data', header=None)
    # columns = np.array(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type'])
    columns = np.array(['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类型'])
    data.rename(columns=dict(list(zip(np.arange(5), columns))), inplace=True) #zip 函数能打包成1match1的元祖
    data['类型'] = pd.Categorical(data['类型']).codes
    print(data.head(5))
    x = data[columns[:-1]]
    y = data[columns[-1]]

    if stype == 'pca':
        pca = PCA(n_components=2, whiten=True, random_state=0)
        x = pca.fit_transform(x)
        print('各方向方差：', pca.explained_variance_)
        print('方差所占比例：', pca.explained_variance_ratio_)
        x1_label, x2_label = '组分1', '组分2'
        title = '鸢尾花数据PCA降维'
    if stype == 'tsne':
        tsne = TSNE(n_components =2 ,learning_rate=100)
        x = tsne.fit_transform(x)
        x1_label, x2_label = '组分1', '组分2'
        title = '鸢尾花数据TSNE降维'
    if stype == 'SelectKBest':
        fs = SelectKBest(chi2, k=2)
        # fs = SelectPercentile(chi2, percentile=60)
        fs.fit(x, y)
        print('fs',fs)
        idx = fs.get_support(indices=True)
        print('分数',fs.scores_)
        print('fs.get_support() = ', idx)
        # print('x' , x)
        # print(x.index)
        x = x.iloc[:,2:4]
        print(x)
        x = x.values    # 为下面使用方便，DataFrame转换成ndarray
        x1_label, x2_label = columns[idx]
        title = '鸢尾花数据特征选择'
    print(x[:5])
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, marker='o', cmap=cm_dark)
    plt.grid(b=True, ls=':', color='k')
    plt.xlabel(x1_label, fontsize=12)
    plt.ylabel(x2_label, fontsize=12)
    plt.title(title, fontsize=15)
    # plt.savefig('1.png')
    plt.show()

    x, x_test, y, y_test = train_test_split(x, y, train_size=0.7)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=True)),
        ('lr', LogisticRegressionCV(Cs=np.logspace(-3, 4, 8), cv=5, fit_intercept=False,penalty='l2'))
    ])
    model.fit(x, y)
    print('最优参数：', model.get_params('lr')['lr'].C_)
    y_hat = model.predict(x)
    print('训练集精确度：', metrics.accuracy_score(y, y_hat))
    y_test_hat = model.predict(x_test)
    print(type(y_test_hat))
    print('测试集精确度：', metrics.accuracy_score(y_test, y_test_hat))

    N, M = 500, 400     # 横纵各采样多少个值
    x1_min, x1_max = extend(x[:, 0].min(), x[:, 0].max())   # 第0列的范围
    x2_min, x2_max = extend(x[:, 1].min(), x[:, 1].max())   # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点;生成网格型数据，可以接受两个一维数组生成两个二维矩阵
    #生成的是：t1(len(t1）*len(t2))、t2（len(t2)*len(t1)）竖着的
    x_show = np.stack((x1.flat, x2.flat), axis=1)   # 测试点,从N*M 变成M*N
    y_hat = model.predict(x_show)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示,就是颜色分类那个，这个函数可以展示值的分区
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, edgecolors='k', cmap=cm_dark)  # 样本的显示
    plt.xlabel(x1_label, fontsize=12)
    plt.ylabel(x2_label, fontsize=12)
    plt.xlim(x1_min, x1_max)#坐标轴长度
    plt.ylim(x2_min, x2_max)
    plt.grid(b=True, ls=':', color='k')
    # 画各种图
    # a = mpl.patches.Wedge(((x1_min+x1_max)/2, (x2_min+x2_max)/2), 1.5, 0, 360, width=0.5, alpha=0.5, color='r')
    # plt.gca().add_patch(a)
    patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
              mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
              mpatches.Patch(color='#A0A0FF', label='Iris-virginica')] #形成彩色片区
    plt.legend(handles=patchs, fancybox=True, framealpha=0.8, loc='lower right')
    plt.title('鸢尾花Logistic回归分类效果', fontsize=15)
    plt.show()
