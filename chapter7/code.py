# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("white")
sns.set_context("notebook")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题
import os
os.chdir(os.getcwd())
"""
programmer_1-->关于原始数据的一些特征描述并保存为新表，使用describe函数
programmer_2-->对原始数据进行清理，对其中某些数据做运算，并进行保存
programmer_21() --> 获得样本数据,即 LRFMC 模型数据
programmer_3-->标准化数据并进行保存
programmer_4-->使用KMeans对数据进行聚类分析
"""
if not os.path.exists('./img'):
    os.mkdir('./img')


def programmer_1(datafile = 'data/air_data.csv',
    resultfile = 'tmp/explore.xls'):

    data = pd.read_csv(datafile, encoding='utf-8')

    # 包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）；T是转置，转置后更方便查阅
    explore = data.describe(percentiles=[], include='all').T
    # describe()函数自动计算非空值数，需要手动计算空值数
    explore['null'] = len(data) - explore['count']

    explore = explore[['null', 'max', 'min']]
    explore.columns = [u'空值数', u'最大值', u'最小值']
    '''这里只选取部分探索结果。
    describe()函数自动计算的字段有count（非空值数）、unique（唯一值数）、top（频数最高者）、freq（最高频数）、mean（平均值）、std（方差）、min（最小值）、50%（中位数）、max（最大值）'''

    explore.to_excel(resultfile)


def programmer_2(datafile = 'data/air_data.csv',
    cleanedfile = 'tmp/data_cleaned.csv'):

    data = pd.read_csv(datafile, encoding='utf-8')

    # 使用乘法运算非空数值的数据，因为numpy不支持*运算，在这里换做&运算
    data = data[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull()]

    # 只保留票价非零的，或者平均折扣率与总飞行公里数同时为0的记录。
    index1 = data['SUM_YR_1'] != 0
    index2 = data['SUM_YR_2'] != 0
    index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)  # 该规则是“与”
    data = data[index1 | index2 | index3]  # 该规则是“或”

    data.to_csv(cleanedfile)
    
def programmer_21(datafile = 'tmp/data_cleaned.csv',
    zscorefile='data/zscoredata_01.xls'):
    '''数据处理程序.'''
    
    data=pd.read_csv(datafile,engine='python',index_col=0,encoding='utf-8')
    data=data[['LOAD_TIME','FFP_DATE','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
    # 计算 LRFMC
    import datetime
    data['L']=pd.Series(map(lambda x,y:round((datetime.datetime.strptime(x,'%Y/%m/%d')-
        datetime.datetime.strptime(y,'%Y/%m/%d')).days/30.0+0.0000000001,2),data['LOAD_TIME'],data['FFP_DATE']))
    data['LAST_TO_END']=data['LAST_TO_END'].apply(lambda x:round(x/30.0+0.0000000001,2))
    data.drop(['LOAD_TIME','FFP_DATE'],axis=1,inplace=True)
    data.columns=['R','F','M','C','L']
    data=data[['L','R','F','M','C']]
    data.fillna(0,inplace=True)
    
    data.to_excel(zscorefile,index=False)

def programmer_3(datafile = 'data/zscoredata_01.xls',
    zscoredfile = 'tmp/zscoreddata_01.xls'):

    data = pd.read_excel(datafile)
    # 核心语句，实现标准化变换，类似地可以实现任何想要的变换。
    data = (data - data.mean(axis=0)) / (data.std(axis=0))
    data.columns = ['Z' + i for i in data.columns]
    data.fillna(0,inplace=True)

    data.to_excel(zscoredfile, index=False)
    
# 保存和导入模型
 
#保存模型
def save_model(model, filepath):
    joblib.dump(model, filename=filepath)

def load_model(filepath):
    model = joblib.load(filepath)
    return model

def programmer_4(inputfile = 'tmp/zscoreddata_01.xls',k=5,load=False):
    data = pd.read_excel(inputfile)

    if load:  # 导入本地模型
        kmodel=load_model('kmeans.m')
    else:
        kmodel = KMeans(n_clusters=k, n_jobs=4)
    kmodel.fit(data)

    center=pd.DataFrame(kmodel.cluster_centers_,columns=['ZL','ZR','ZF','ZM','ZC'])
    labels=pd.Series(kmodel.labels_).value_counts()
    labels.name='cluster_num'
    data=center.join(labels)
    data['cluster_names']=['customer%s'%i for i in data.index]
    data=data[['cluster_names','cluster_num','ZL','ZR','ZF','ZM','ZC']]
    print(data)
    
    # 保存模型
    save_model(kmodel,'kmeans.m')
    return data,kmodel.labels_

# 绘制雷达图

def plot_radar(data,title=''):
    '''
    the first column of the data is the cluster name;
    the second column is the number of each cluster;
    the last are those to describe the center of each cluster.
    '''
    kinds = data.iloc[:, 0]
    labels = data.iloc[:, 2:].columns
    centers = pd.concat([data.iloc[:, 2:], data.iloc[:,2]], axis=1)
    centers = np.array(centers)
    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True) # 设置坐标为极坐标
    
    # 画若干个五边形
    floor = np.floor(centers.min())     # 大于最小值的最大整数
    ceil = np.ceil(centers.max())       # 小于最大值的最小整数
    for i in np.arange(floor, ceil + 0.5, 0.5):
        ax.plot(angles, [i] * (n + 1), '--', lw=0.5 , color='black')
    
    # 画不同客户群的分割线
    for i in range(n):
        ax.plot([angles[i], angles[i]], [floor, ceil], '--', lw=0.5, color='black')
    
    # 画不同的客户群所占的大小
    for i in range(len(kinds)):
        ax.plot(angles, centers[i], lw=2, label=kinds[i])
        #ax.fill(angles, centers[i])
    
    ax.set_thetagrids(angles * 180 / np.pi, labels) # 设置显示的角度，将弧度转换为角度
    plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0.0)) # 设置图例的位置，在画布外
    
    ax.set_theta_zero_location('N')        # 设置极坐标的起点（即0°）在正北方向，即相当于坐标轴逆时针旋转90°
    ax.spines['polar'].set_visible(False)  # 不显示极坐标最外圈的圆
    ax.grid(False)                         # 不显示默认的分割线
    ax.set_yticks([])                      # 不显示坐标间隔
    
    plt.savefig('img/聚类结果属性雷达图%s.png'%title,dpi=200)
    
# 属性直方图
def cluster_density(data,datafile,labels,k=5,title=''):
    '''绘制每一个聚类结果各个属性的分布图.'''
    if datafile:
        #datafile = 'data/zscoredata_01.xls'
        data=pd.read_excel(datafile)
    r=pd.Series(labels)
    r.name='cluster_names'
    #data['cluster']=labels
    def density_plot(data, k,title):
        p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False,grid=True)
        [p[i].set_ylabel('密度') for i in range(k)]
        p[0].set_title(title)
        [p[i].grid(linestyle='--',alpha=0.8) for i in range(k)]
        plt.legend()
        plt.tight_layout()
        return plt
    
    # 保存概率密度图
    for i in range(k):
        density_plot(data[r == i],
                     k,'分群_%s密度图'%(i+1)).savefig('./img/分群_%i%s.png' % ((i+1),title),dpi=200)
        
# 可视化聚类结果        
def programmer_5(data_zs,datafile,r,k=5,title=''):
    # 进行数据降维
    if datafile:
        #datafile='tmp/zscoreddata_01.xls'
        data_zs=pd.read_excel(datafile)
    tsne = TSNE()
    tsne.fit_transform(data_zs)
    tsne = pd.DataFrame(tsne.embedding_, index=data_zs.index)

    # 不同类别用不同颜色和样式绘图
    for i in range(k):
        d = tsne[r == i]
        plt.plot(d[0], d[1])
#    d = tsne[r == 1]
#    plt.plot(d[0], d[1], 'go')
#    d = tsne[r == 2]
#    plt.plot(d[0], d[1], 'b*')
    plt.title('聚类效果图')
    plt.savefig('img/聚类效果图%s.png'%title,dpi=200)
    
def programmer_6(data,datafile,center,labels,threshold=2,k=5,annotate_=True,title=''):
    """
    k：聚类中心数
    threshold：离散点阈值
    iteration：聚类最大循环次数
    """
    if datafile:
        #datafile='tmp/zscoreddata_01.xls'
        data_zs=pd.read_excel(datafile)
    data_zs['cluster']=labels

    norm = []
    for i in range(k):  # 逐一处理
        norm_tmp = data_zs[['ZL', 'ZR', 'ZF','ZM','ZC']][data_zs['cluster'] == i] - center.loc[i,['ZL', 'ZR', 'ZF','ZM','ZC']]
        norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)
        # 求相对距离并添加
        norm.append(norm_tmp / norm_tmp.median())

    norm = pd.concat(norm)
    # 正常点
    norm[norm <= threshold].plot(style='go')
    # 离群点
    discrete_points = norm[norm > threshold]
    discrete_points.plot(style='rx')
    # 标记离群点
    if annotate_:
        for i in range(len(discrete_points)):
            _id = discrete_points.index[i]
            n = discrete_points.iloc[i]
            plt.annotate('(%s, %0.2f)' % (_id, n), xy=(_id, n), xytext=(_id, n))
    else:
        pass

    plt.xlabel('编号')
    plt.ylabel('相对距离')
    plt.title('离群点标记(%d倍标准差)'%threshold)
    plt.grid(linestyle='--',alpha=0.8)
    plt.savefig('img/离群点标记%s.png'%title,dpi=200)
    
    # 导出文件清除了离群值后的数据记录
    #data_zs[['ZL', 'ZR', 'ZF','ZM','ZC']][norm <= threshold].to_excel('tmp/zscoreddata_01_%d.xls'%threshold,index=False)
    
    return norm.index


if __name__ == "__main__":
#    programmer_1()    # describe
#    programmer_2()    # 清洗数据
#    programmer_21()   # 生成样本数据
#    programmer_3()    # 标准化
    center,labels=programmer_4()    # 聚类
    plot_radar(center)              # 绘制雷达图
    cluster_density(None,datafile='data/zscoredata_01.xls',labels=labels) # 绘制每一个聚类结果的属性分布图
    programmer_5(None,datafile='tmp/zscoreddata_01.xls',r=labels)
    # 离群点
    norm_index=programmer_6(None,datafile='tmp/zscoreddata_01.xls',center=center,labels=labels,
                      threshold=3,annotate_=False)
    # 读取数据，清除异常记录
    print('去除利群值影响.')
    data=pd.read_excel('data/zscoredata_01.xls').loc[norm_index,:] 
    data.to_excel('data/zscoredata_01_3.xls',index=False)  # 去除离群值的影响
    # 标准化
    programmer_3(datafile = 'data/zscoredata_01_3.xls',
    zscoredfile = 'tmp/zscoreddata_01_3.xls')    # 标准化
    center,labels=programmer_4(inputfile = 'tmp/zscoreddata_01_3.xls',load=True)    # 聚类
    plot_radar(center,title='(去异常值)')              # 绘制雷达图
    cluster_density(None,datafile='data/zscoredata_01_3.xls',labels=labels,title='(去异常值)') # 绘制每一个聚类结果的属性分布图
    programmer_5(None,datafile='tmp/zscoreddata_01_3.xls',r=labels,title='(去异常值)')
    norm_index=programmer_6(None,datafile='tmp/zscoreddata_01_3.xls',center=center,labels=labels,
                      threshold=3,title='(去异常值)')
    # 
    pass
