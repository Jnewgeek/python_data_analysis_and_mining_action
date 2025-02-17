# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:42:13 2017

@author: wnma3
"""

import os 
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import pandas as pd
import pywt
from pandas import DataFrame, Series
from scipy.interpolate import lagrange
from scipy.io import loadmat  # mat是MATLAB的专用格式，调用loadmat方法读取
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
"""
代码说明：
ployinterp_column-->拉格朗日填充数值

修改拉格朗日插值法函数,原始函数结果与《Python数据分析与挖掘》差异较大,
且有错误值.

programmer_1-->筛选异常数据（包括NaN）进行填充
programmer_2-->最小-最大规范化、零-均值规范化、小数定标规范化
programmer_4-->基本的dataframe操作
programmer_5-->利用小波分析（？？？）进行特征分析
programmer_6-->利用PCA计算特征向量，用于降维分析
"""
path = os.getcwd()

for i in ['./data','./img','./tmp']:
    if not os.path.exists(i):
        os.mkdir(i)

def programmer_1(method=1):
    inputfile = path + '/data/catering_sale - 副本.xls'
    outputfile = path + '/tmp/sales.xls'

    data = pd.read_excel(inputfile)
    
    # 重新排序
    data=data.sort_values(by='日期').reset_index(drop=True)
    
    # 新增一列对比原始数据
    data['销量_new']=data['销量']

    data['销量_new'][(data['销量_new'] < 400) | (data['销量_new'] > 5000)] = None
    
    data['销量']=data['销量_new']

    if method==1:
        def ployinterp_column(n, s, k=5):
            y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数
            y = y[y.notnull()]  # 剔除空值
            return lagrange(y.index, list(y))(n)  # 插值并返回插值结果
    else:
        def ployinterp_column(n, s, k=5):
            y = s[list(range(n - k,n + 1 + k))]
            y = y[y.notnull()]
            return lagrange([i-(n-k-1) for i in y.index], list(y))(k+1)

    df = data[data['销量_new'].isnull()]

    for index in df.index:
        data['销量_new'][index] = ployinterp_column(index, data['销量_new'])

    data.to_excel(outputfile,index=False)
    
    return data[data['销量'].isnull()]


def programmer_2():
    datafile = path + '/data/normalization_data.xls'
    data = pd.read_excel(datafile, header=None)

    print((data - data.min()) / (data.max() - data.min()))
    print((data - data.mean()) / data.std())
    print(data / 10**np.ceil(np.log10(data.abs().max())))


# 聚类画图
def programmer_3():
    datafile = path + '/data/discretization_data.xls'
    data = pd.read_excel(datafile)
    data = data[u'肝气郁结证型系数'].copy()
    k = 4

    # 方法一， 直接对数组进行分类
    d1 = pd.cut(data, k, labels=range(k))

    # 方法二， 等频率离散化
    w = [1.0 * i / k for i in range(k + 1)]
    # percentiles表示特定百分位数，同四分位数
    w = data.describe(percentiles=w)[4:4 + k + 1]
    w[0] = w[0] * (1 - 1e-10)   # 区间原因所以要把最小值再缩小一点
    d2 = pd.cut(data, w, labels=range(k))

    #　方法三，使用Kmeans
    kmodel = KMeans(n_clusters=k, n_jobs=4)

    kmodel.fit(data.values.reshape(len(data), 1))
    # 输出聚类中心，并且排序
    c = DataFrame(kmodel.cluster_centers_).sort_values(0)

    # 相邻两项求中点，作为边界点
    w = DataFrame.rolling(c, 2).mean().iloc[1:]
    # 加上首末边界点
    w = [0] + list(w[0]) + [data.max()]
    d3 = pd.cut(data, w, labels=range(k))

    def cluster_plot(d, k,title):
        plt.figure(figsize=(8, 3))
        for j in range(0, k):
            plt.plot(data[d == j], [j for i in d[d == j]], 'o')
        plt.ylim(-0.5, k - 0.5)
        plt.title(title)
        plt.savefig('./img/%s.png'%title,dpi=200)
        #return plt

    cluster_plot(d1, k,'等宽离散化结果')
    cluster_plot(d2, k,'等频离散化结果')
    cluster_plot(d3, k,'聚类离散化结果')


def programmer_4():
    inputfile = path + "/data/electricity_data.xls"
    outputfile = path + "/tmp/electricity_data.xls"
    data = pd.read_excel(inputfile)
    data[u"线损率"] = (data[u"供入电量"] - data[u"供出电量"]) / data[u"供入电量"]
    data.to_excel(outputfile, index=False)


def programmer_5():
    inputfile = path + "/data/leleccum.mat"
    mat = loadmat(inputfile)
    signal = mat["leleccum"][0]
    """
    处理数据
    返回结果为level+1个数字
    第一个数组为逼近系数数组
    后面的依次是细节系数数组    
    """
    coeffs = pywt.wavedec(signal, "bior3.7", level=5)
    print(coeffs)


def programmer_6(n=None):
    inputfile = path + "/data/principal_component.xls"
    outputfile = path + "/tmp/dimention_reducted.xls"

    data = pd.read_excel(inputfile, header=None)
    
    if n==None:
        pca = PCA()
        pca.fit(data)
        # 返回各个模型的特征向量
        print(pca.components_)
        # 返回各个成分各自的方差百分比
        print(pca.explained_variance_ratio_)
        data.to_excel(outputfile, index=False,header=False)
    else:
        pca=PCA(n)
        pca.fit(data)
        # 返回各个模型的特征向量
        low_d=pca.transform(data)
        # 返回各个成分各自的方差百分比
        pd.DataFrame(low_d).to_excel(outputfile,index=False,header=False)
        print(pca.inverse_transform(low_d))   # 复原数据


if __name__ == '__main__':
    data1=programmer_1(method=1)
    data2=programmer_1(method=2)
#    programmer_2()
#    programmer_3()
#    programmer_4()
#    programmer_5()
#    programmer_6()
#    programmer_6(3)
    #pass
