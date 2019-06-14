#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W1401
"""
Created on Mon Nov  6 21:04:24 2017

@author: lu
"""

import numpy as np

import pandas as pd
from sqlalchemy import create_engine
import time
import datetime

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

import threading
from time import ctime
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(os.getcwd())

if not os.path.exists('./img'):
    os.mkdir('img')

"""
这部分代码主要是用Python连接数据库，提取数据进行分析。
所j以在运行代码之前需要讲sql语句运行一遍将数据插入到mysql数据库中
注意这里需要提前创建一个database，并且在开头增加使用database的语句
mysql -uroot -p < 7law.sql
需要等待一会

此部分代码没有运行，存在一定问题

count107-->统计107类别情况
programmer_1-->大概了解了处理数据意图
programmer_2-->提取所需数据，并且保存到数据库中
programmer_3-->进行数据筛选，保存到数据库中
programmer_4-->合并某些特征为一个特征，保存到数据库
programmer_5-->推荐矩阵

MyThread --> 新增多线程类,加快数据分布的实现
"""
# 多线程
class MyThread(threading.Thread):

	def __init__(self,func,args,name='',prints=False):
		threading.Thread.__init__(self)
		self.name=name
		self.func=func
		self.args=args
		self.prints=prints

	def getResult(self):
		return self.res

	def run(self):
		if self.prints:print('Starting < %s > at: %s\n'%(self.name,ctime()))
		self.res=self.func(*self.args)
		if self.prints:print('< %s > finished at: %s\n'%(self.name,ctime()))
        
def get_engine():
    '''
    数据库地址.
    '''
    engine = create_engine(
        "mysql+pymysql://root:123456@127.168.1.162:3306/law?charset=utf8")
    return engine
        
# 统计 107 内容
def count107(i):
    j = i[["fullURL"]][i["fullURLId"].str.contains("107")].copy()
    # 添加空列
    j["type"] = None
    #  利用正则进行匹配，并重命名
    j["type"][j["fullURL"].str.contains("info/.+?/")] = u"知识首页"
    j["type"][j["fullURL"].str.contains("info/.+?/.+?")] = u"知识列表页"
    j["type"][j["fullURL"].str.contains("/\d+?_*\d+?\.html")] = u"知识内容页"
    return j["type"]

# 分块读取数据
def read_sql():
    print(">> 开始分块读取数据...")
    
    time1=time.time()
    sql = pd.read_sql("all_gzdata",get_engine(), chunksize=10000)
    time2=time.time()
    print('>> 读取用时%.2fs'%(time2-time1))
    # 连接数据
    sql=pd.concat(sql)
    return sql

# 保存到数据库中
def to_sql(sql,title,tablename):
    '''
    将当天的统计数据保存到数据库中.
    '''
    # 添加时间字段
    sql['update_time']=datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')
    print('>> 保存< %s >到数据库中...'%title)
    time1=time.time()
    sql.to_sql(tablename, get_engine(), index=False, if_exists="append")
    time2=time.time()
    print('>> 用时%.2fs'%(time2-time1))

# 查看数据分布并绘图
def value_count(counts,column=['网页类型','记录数'],per='百分比',title='网页类别统计',special=False,N=None):
        '''
        根据传入的列进行汇总统计,查看占比情况,并绘图.
        special:特殊绘制,记录数和用户数在同一张图上展示,默认为False
        '''
        counts = counts.value_counts().reset_index()
        # 自动重新设置index并将原来的index作为columns
        # counts = counts.reset_index()
        counts.columns = column
        # 计算百分比
        counts[per]=counts[column[1]]/counts[column[1]].sum()
        if special==False: # 如果是同一个指标,则绘制帕累托曲线,否则绘制另一个指标的曲线
            p=1.0*counts[column[1]].cumsum()/counts[column[1]].sum()   # 累计百分比
            index=len(p)-(p>=0.8).sum()
        else:
            # 计算记录百分比
            counts['记录百分比']=pd.Series(map(lambda x,y:x*y/N if str(x).isdigit() else np.nan,counts[column[0]],counts[column[1]]))
            # 填补缺失值
            counts['记录百分比'].fillna(1-counts['记录百分比'].sum(),inplace=True)
            p=counts['记录百分比']
        # 计算分位数的点位置以及记录长度
        length=len(counts)
        # 修改列名，提取每个列名前三个数字，用到了正则表达式
        # 按类别排序
        print('%s\n--------------------------------'%title)
        print(counts)
        
        # 绘图
        #plt.figure()
        counts[column].set_index(column[0]).plot(kind='bar')
        plt.xticks(rotation=0)
        plt.ylabel('用户数(个)' if special else '记录数(个)')
        p.plot(color='r', secondary_y=True, style='-o', linewidth=2)
        if special or index==0:  # 如果是第一个超过0.8或者是两个指标的图则不显示箭头注释
            pass
        else:
            plt.annotate(
                format(p[index], '.4%'),
                xy=(index, p[index]),
                xytext=(index * 0.95, p[index] * 0.95),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.ylabel('记录占比(%)' if special else '累计占比(%)')
        # 调整X轴坐标
        plt.xlim([-0.5,length-0.5])
        plt.title(title)
        plt.grid(axis='y',linestyle='--',alpha=0.8)
        # 保存图形
        plt.savefig('img/%s.png'%title,dpi=200)
        
        # 传回数据
        return counts

# 查看数据分布主函数
def programmer_1(sql):
    """
    用pymysql连接本地数据库
    按个人情况进行更改连接语句
    engine表示连接数据的引擎，chunksize表示每次读取数据量
    此时‘sql’只是一个容器
    """
    # (1) 网页类型,提取前3位
    sql['网页类型'] = sql['fullURLId'].str.extract("(\d{3})")
    # 绘制分布图
    counts_1=value_count(sql['网页类型'],column=['网页类型','记录数'],title='网页类别统计')
    # 保存数据
    to_sql(counts_1,'网页类别统计','statistic_page_category')
    
    # (2) 咨询类别内部统计
    counts_2=sql[sql['网页类型']==counts_1['网页类型'][0]]['fullURLId']
    counts_2=value_count(counts_2,column=['101开头类型','记录数'],title='101开头类别统计')
    # 保存数据
    to_sql(counts_2,'101开头类别统计','statistic_101_head')
    
    # (3) 知识类型内部统计
    counts_3 = value_count(count107(sql),column=['107开头类型','记录数'],title='知识类型内部统计')
    
    # (4) 带问号字符网址类型统计
    counts_4=sql[sql['fullURL'].str.contains("\?")]['fullURLId']
    counts_4=value_count(counts_4,column=['网页ID','记录数'],title='带问号字符网址类型统计')
    
    # (5) 其他类型统计表
    counts_5=sql[(sql['fullURLId']==counts_4['网页ID'][0])&(sql['fullURL'].str.contains('\?'))]['pageTitle']
    # 数据清洗
    def clean_title(x):
        if '快车-律师助手' in x:
            return '快车-律师助手'
        elif '发布法律咨询' in x:
            return '免费发布咨询'
        elif '咨询发布成功' in x:
            return '咨询发布成功'
        elif '快搜' in x:
            return '快搜'
        else:
            return '其他类型'
    # 清洗数据
    counts_5=counts_5.apply(clean_title)
    # 其他类型统计(1999001)
    counts_5=value_count(counts_5,column=['网页标题','记录数'],title='其他类型统计(%s)'%counts_4['网页ID'][0])

    # (6)用户点击次数统计
    counts_6 = sql["realIP"].value_counts().apply(lambda x:x if x<=7 else '7次以上')
    counts_6=value_count(counts_6,column=['点击次数','用户数'],per='用户百分比',title='用户点击次数统计',special=True,N=len(sql))
    # 保存数据
    counts_6['点击次数']=counts_6['点击次数'].apply(lambda x:'%s'%x if str(x).isdigit() else '>%s'%x.replace('次以上',''))
    to_sql(counts_6,'用户点击次数统计','statistic_user_clicks')
    

def programmer_2(sql):
    time1=time.time()
    d = sql[["realIP", "fullURL"]].drop_duplicates().reset_index(drop=True).copy()
    d = d[d["fullURL"].str.contains("\.html")].copy()
    #d['fullURL']=d['fullURL'].apply(lambda x:x.encode('utf-8').decode('utf-8'))
    d.to_sql("cleaned_gzdata", get_engine(), index=False, if_exists="append")
    time2=time.time()
    print('finished!用时: %.2fs'%(time2-time1))


def programmer_3():
    time1=time.time()
    sql=pd.read_sql('cleaned_gzdata',get_engine(),chunksize=10000)
    for i in sql:
        d = i.copy()
        # 替换关键词
        d["fullURL"] = d["fullURL"].str.replace("_\d{0,2}.html", ".html")
        # 去除重复数据
        d = d.drop_duplicates()
        d.to_sql("changed_gzdata", get_engine(), index=False, if_exists="append")
    time2=time.time()
    print('finished!用时: %.2fs'%(time2-time1))


def programmer_4():
    time1=time.time()
    sql=pd.read_sql('changed_gzdata',get_engine(),chunksize=10000)
    for i in sql:
        d = i.copy()
        d["type_1"] = d["fullURL"]
        d["type_1"][d["fullURL"].str.contains("(ask)|(askzt)")] = "zixun"
        d.to_sql("splited_gzdata", get_engine(), index=False, if_exists="append")
    time2=time.time()
    print('finished!用时: %.2fs'%(time2-time1))


def Jaccard(a, b):
    return 1.0 * (a * b).sum() / (a + b - a * b).sum()


def programmer_5(tablename):
    # 读取数据
    tablename='splited_gzdata'
    sql=pd.read_sql(tablename,get_engine(),chunksize=10000)
    data=pd.concat(sql)
    data=data[(data['fullURL'].str.contains('hunyin'))&(data['type_1']!='zixun')][['realIP','fullURL']]   # 获取模型数据
    # 加载随机打乱数据顺序的随机数
    from sklearn.model_selection import train_test_split
    data_train, data_test= train_test_split(data, test_size=0.1)
    print('>> 数据集准备完成.')
    print('''训练数据总数: %d
    物品个数: %d
    访问平均次数: %d
    测试数据总数: %d       
    '''%(len(data_train),len(data_train['fullURL'].unique()),
    len(data_train)/len(data_train['fullURL'].unique()),len(data_test)))
        
    class Recommender:

        sim = None
        
        def __init__(self,data_train,data_test):
            # 获取数据
            self._data_train=data_train.set_index('realIP')
            self._data_test=data_test.set_index('realIP')
            # 用户名
            self._ip_train,self._url_train=data_train.realIP.unique(),data_train.fullURL.unique()
            self._ip_test,self._url_test=data_test.realIP.unique(),data_test.fullURL.unique()
            # 生成稀疏矩阵
            self._train_array=pd.crosstab(data_train['realIP'],data_train['fullURL']).values
            self._test_array=pd.crosstab(data_test['realIP'],data_test['fullURL']).values
            
        # 判断距离（相似性）
        def similarity(self, x, distance):
            
            y = np.ones((len(x), len(x)))
            for i in range(len(x)):
                for j in range(len(x)):
                    y[i, j] = distance(x[i], x[j])

            return y

        def fit(self, x, distance=Jaccard):
            self.sim = self.similarity(x, distance)
        
        # 推荐矩阵
        def recommend(self, a):
            return np.dot(self.sim, a) * (1 - a)

    test=Recommender(data_train,data_test)  # 将数据导入模型
    test.fit(test._train_array)
    
    

if __name__ == "__main__":
    # 读取数据
    sql=read_sql()
    programmer_1(sql.copy())     # 给个维度的数据分布
    programmer_2(sql.copy())     # 筛选后缀为.html的网址
    programmer_3()               # 合并
    programmer_4()               # 分割
    programmer_5()
    pass
