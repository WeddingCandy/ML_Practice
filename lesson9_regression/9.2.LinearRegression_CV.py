#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge,LinearRegression,ElasticNetCV,LassoCV
from sklearn.model_selection import GridSearchCV



if __name__ == "__main__":
    # pandas读入
    data = pd.read_csv('Advertising.csv')    # TV、Radio、Newspaper、Sales
    print(data)
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    print(x)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
    model1 = Lasso()
    model2 = LassoCV()
    model3 = Ridge()
    model4 = LinearRegression()
    model5 = ElasticNetCV()
    alpha_can = np.logspace(-3, 2, 10) #参数范围
    np.set_printoptions(suppress=True)
    print('alpha_can = ', alpha_can)
    lasso_model = GridSearchCV(model1, param_grid={'alpha': alpha_can}, cv=5)#cv折数，model模型，param_grid参数范围
    lasso_model.fit(x_train, y_train)
    lassocv_model = model2.fit(x_train,y_train)
    ridge_model = model3.fit(x_train,y_train)
    lr_model = model4.fit(x_train,y_train)
    encv_model = model5.fit(x_train,y_train)
    print('lasso超参数：\n', lasso_model.best_params_)
    # print('lassocv超参数：\n', lassocv_model.best_params_)
    # print('ridge超参数：\n',ridge_model.best_params_)
    # print('lr超参数：\n',lr_model.best_params_)
    # print('ElasticNetCV超参数：\n',encv_model.best_params_)


    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_hat_lasso = lasso_model.predict(x_test)
    y_hat_lassocv = lassocv_model.predict(x_test)
    y_hat_ridge = ridge_model.predict(x_test)
    y_hat_encv = encv_model.predict(x_test)
    y_hat_lr = lr_model.predict(x_test)
    print(lasso_model.score(x_test, y_test))
    mse_lasso = np.average((y_hat_lasso - np.array(y_test)) ** 2)  # Mean Squared Error
    mse_lassocv = np.average((y_hat_lassocv - np.array(y_test)) ** 2)  # Mean Squared Error
    mse_ridge = np.average((y_hat_ridge - np.array(y_test)) ** 2)  # Mean Squared Error
    mse_lr = np.average((y_hat_lr - np.array(y_test)) ** 2)  # Mean Squared Error
    mse_encv = np.average((y_hat_encv - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse_lasso = np.sqrt(mse_lasso)  # Root Mean Squared Error
    print(mse_lasso, rmse_lasso)
    print('最小均方误差', min(mse_encv,mse_lasso,mse_lassocv,mse_lr,mse_ridge))
    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label='真实数据')
    plt.plot(t, y_hat_lasso, 'g-', linewidth=3, label='lasso预测数据')
    plt.plot(t,y_hat_ridge, 'y-',linewidth=1, label='ridge预测数据')
    plt.plot(t, y_hat_encv, 'bo', linewidth=1, label='encv预测数据')
    # plt.plot(t, y_hat_lassocv, 'd-', linewidth=1, label='lassocv预测数据')
    plt.title('线性回归预测销量', fontsize=18)
    plt.legend(loc='upper left')
    plt.grid(b=True, ls=':')
    plt.show()
