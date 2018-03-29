# !/usr/bin/python
# -*- coding:gbk -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == "__main__":
    stock_max, stock_min, stock_close, stock_amount = np.loadtxt('SH600000.txt', delimiter='\t', usecols=(2, 3, 4, 5), unpack=True)
    N = 100
    stock_close = stock_close[:N]
    print(stock_close)

    n = 10
    weight = np.ones(n)
    weight /= weight.sum()
    print(weight)
    stock_sma = np.convolve(stock_close, weight, mode='valid')  # simple moving average

    weight = np.linspace(1, 0, n)
    weight = np.exp(weight)
    weight /= weight.sum()
    print(weight)
    stock_ema = np.convolve(stock_close, weight, mode='valid')  # exponential moving average

    t = np.arange(n-1, N)
    poly = np.polyfit(t, stock_ema, 10)
    print(poly)
    stock_ema_hat = np.polyval(poly, t)

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(np.arange(N), stock_close, 'ro-', linewidth=2, label='ԭʼ���̼�')
    t = np.arange(n-1, N)
    plt.plot(t, stock_sma, 'b-', linewidth=2, label='���ƶ�ƽ����')
    plt.plot(t, stock_ema, 'g-', linewidth=2, label='ָ���ƶ�ƽ����')
    plt.legend(loc='upper right')
    plt.title('��Ʊ���̼��뻬��ƽ����MA', fontsize=15)
    plt.grid(True)
    plt.show()

    print(plt.figure(figsize=(8.8, 6.6), facecolor='w'))
    plt.plot(np.arange(N), stock_close, 'ro-', linewidth=1, label='ԭʼ���̼�')
    plt.plot(t, stock_ema, 'g-', linewidth=2, label='ָ���ƶ�ƽ����')
    plt.plot(t, stock_ema_hat, '-', color='#FF4040', linewidth=3, label='ָ���ƶ�ƽ���߹���')
    plt.legend(loc='upper right')
    plt.title('����ƽ����MA�Ĺ���', fontsize=15)
    plt.grid(True)
    plt.show()
