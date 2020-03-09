import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv(r'.\geekDA\data23.csv')
#需工作的最近根文件目录，不区分大小写
#相对路径，编辑launch.json，在configuration中使用参数"cwd":"${fileDirname}" 调试模式可以，普通运行报错

pd.set_option('display.max_columns',None) #把折叠的列显示出来，none=不限制列数
""" print ('describe' + '-'*30)
print (data.describe())
print ('info' + '-'*30)
print (data.info())
print ('head' + '-'*30)
print (data.head(5))
print ('columns' + '-'*30)
print (data.columns) """

features_mean = list(data.columns[2:12])
features_se= list(data.columns[12:22])
features_worse = list (data.columns[22:32])
data.drop(data.columns[0],axis=1,inplace = True) #使用列名无需带前缀'id'，位置：data.columns[1]
data['diagnosis']=data['diagnosis'].map({'M':1,"B":0}) #map 函数
""" print (features_mean)
print (data.head(5)) """

""" map函数替换练习
map_dic = {'key1':'value1', 'key2':'value2'}
print (map_dic)
print (map_dic.keys())
print (map_dic.values()) """

from collections import Counter
print (Counter(data['diagnosis']))

sns.countplot(data['diagnosis'],label="count")
plt.show()
#plt.savefig(r'D:\code\codegit\Python\geekDA\23diacount.png')
#用热力图呈现features_mean字段之间的相关性
corr=data[features_mean].corr() 
# DataFrame.corr(method='', min_periods=1);method：可选值为{‘pearson’, ‘kendall’, ‘spearman’};min_periods：样本最少的数据量
plt.figure(figsize=(14,14))
# annot=True显示每个方格的数据
sns.heatmap(corr,annot=True)
plt.show()
#plt.savefig(r'D:\code\codegit\Python\geekDA\热力图.png')
