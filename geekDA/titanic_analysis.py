import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer #特征提取
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.model_selection import cross_val_score  #K折交叉验证
import graphviz
from sklearn import tree 

#导入数据
train_data = pd.read_csv(r'.\Python\geekDA\train.csv') 
test_data = pd.read_csv(r'.\Python\geekDA\test.csv')

""" 
print ('info' + '-'*26)
print (train_data.info())
print ('decribe' + '-'*26)
print (train_data.describe())
print ('字符串类型' + '-'*26)
print (train_data.describe(include=['O'])) # 是o 不是 0 ，[]可省略
print ('前几行'+ '-'*26)
print (train_data.head())
print ('后几行'+ '-'*26)
print (train_data.tail()) 
"""

#数据清洗
#使用平局值来填充年龄/票价中的nan值
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
""" 
print (train_data.info())
print (test_data.info())
"""
# print (train_data['Embarked'].value_counts())
train_data['Embarked'].fillna('S',inplace= True)
test_data['Embarked'].fillna('S',inplace=True)
#特征选择
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_features = train_data[features]
train_labels = train_data['Survived'] #训练数据
test_features = test_data[features] #测试预测数据

#特征提取 
# DictVectorizer:将可以处理符号化的对象，将符号转成数字 0/1 进行表示
# fit_transform:将特征向量转化为特征值矩阵
dvec = DictVectorizer(sparse = False)
train_features = dvec.fit_transform(train_features.to_dict(orient= 'record'))
# print (dvec.feature_names_)


#构造ID3决策树
clf = DecisionTreeClassifier(criterion= 'entropy') #entropy:ID3;gini:CART
#决策树训练
clf.fit(train_features,train_labels)
test_features = dvec.transform(test_features.to_dict(orient='record'))
#决策树预测
pred_labels = clf.predict(test_features)
#得到决策树准确率
acc_decision_tree = round(clf.score(train_features,train_labels),6)
print (u'score准确率%4lf'%acc_decision_tree)

#使用K折交叉验证，统计决策树准确率
#print (u'cross_val_score%4lf'% np.mean(cross_val_score(clf,train_features,train_labels,cv=10)))
val =np.mean(cross_val_score(clf,train_features,train_labels,cv=10)) # cv 代表对原始数据划分成多少份，也就是 K 值
print ("cross_val_score {value:.4f} ".format(value=val))

#可视化
dot_data = tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(dot_data)
graph.view()
#graph.render(r'E:\code_git\Python\GeekDA\tree')