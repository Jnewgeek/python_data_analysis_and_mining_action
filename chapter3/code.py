# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:03:39 2017

@author: wnma3
"""

import os
#import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import pandas as pd

os.chdir(os.getcwd())
if not os.path.exists('./img'):
    os.mkdir('./img')

"""
代码说明：
programmer_1: 制作箱线图
data.boxplot-->数据转为箱线图的字典格式
plt.annotate-->绘图

programmer_2: 计算数据
range-->极差
var-->方差
dis-->四分距

programmer_3: 画出盈利图（比例和数值）

programmer_4: 计算成对相关性
data.corr()-->dataframe中相互之间的相关性
data.corr()[u'百合酱蒸凤爪'] -->dataframe某一项与其他项的相关性
"""


def programmer_1(file_name):
    catering_sale = file_name
    data = pd.read_excel(catering_sale, index_col=u'日期')

    plt.figure()

    # 画箱线图
    p = data.boxplot(return_type='dict')
    x = p['fliers'][0].get_xdata()
    y = p['fliers'][0].get_ydata()
    y.sort()    # 直接修改原列表,无返回值

    for i in range(len(x)):
        # 处理临界情况， i=0时
        temp = y[i] - y[i - 1] if i != 0 else -78 / 3
        # 添加注释, xy指定标注数据，xytext指定标注的位置（所以需要特殊处理）
        plt.annotate(
            y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.05 - 0.8 / temp, y[i]))
    plt.title('箱型图')
    plt.savefig('img/箱型图.png',dpi=200)


def programmer_2(file_name):
    catering_sale = file_name
    data = pd.read_excel(catering_sale, index_col=u'日期')

    data = data[(data[u'销量'] > 400) & data[u'销量'] < 5000]
    statistics = data.describe()[u'销量']

    statistics['range'] = statistics['max'] - statistics['min']
    statistics['var'] = statistics['std'] / statistics['mean']
    statistics['dis'] = statistics['75%'] - statistics['25%']

    print(statistics)


def programmer_3(file_name):
    dish_profit = file_name  #餐饮菜品盈利数据
    data = pd.read_excel(dish_profit, index_col=u'菜品名')
    data = data[u'盈利'].copy()
    data.sort_values(ascending=False)

    plt.figure()
    data.plot(kind='bar')
    plt.ylabel(u'盈利（元）')
    p = 1.0 * data.cumsum() / data.sum()
    p.plot(color='r', secondary_y=True, style='-o', linewidth=2)
    plt.annotate(
        format(p[6], '.4%'),
        xy=(6, p[6]),
        xytext=(6 * 0.9, p[6] * 0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlim([-0.5,len(data)-0.5])
    plt.ylabel(u'盈利（比例）')
    plt.title('盈利图(值和比例)')
    plt.savefig('img/盈利图(值和比例).png',dpi=200)


def programmer_4(file_name):
    catering_sale = file_name
    data = pd.read_excel(catering_sale, index_col=u'日期')

    print(data.corr())
    print(data.corr()[u'百合酱蒸凤爪'])
    print(data[u'百合酱蒸凤爪'].corr(data[u'翡翠蒸香茜饺']))


if __name__ == "__main__":
#    path = os.getcwd()
    programmer_1('data/catering_sale.xls')
    programmer_2('data/catering_sale.xls')
    programmer_3('data/catering_dish_profit.xls')
    programmer_4('data/catering_sale_all.xls')
    pass
