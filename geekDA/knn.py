from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#加载数据
digits = load_digits()
data = digits.data
#数据探索
""" print ('shape' + '-'*30)
print (data.shape)
print ('image' + '-'*30)
print (digits.images[0])
print ('target' + '-'*30)
print (digits.target[0])
#将第一幅图片打印出来
plt.gray()
plt.imshow(digits.images[0])
plt.show() """

#分割数据
train_x,test_x,train_y,test_y = train_test_split(data,digits.target,test_size=0.25,random_state=33)
#Z-Scroe规范化
ss = preprocessing.StandardScaler()
train_ssx = ss.fit_transform(train_x)
test_ssx = ss.transform(test_x)

#创建knn分类器
knn = KNeighborsClassifier()
knn.fit(train_ssx,train_y)
predict_y = knn.predict(test_ssx)
print ('knn准确率：{value:.4f}'.format(value=accuracy_score(test_y,predict_y)) ) 

#创建svm分类器
svm = SVC(gamma='auto')
""" FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning) """
svm.fit(train_ssx,train_y)
predict_y = svm.predict(test_ssx)
print ('kvm准确率：{value:.4f}'.format(value=accuracy_score(test_y,predict_y)) ) 


#采用Min-Max规范化
mm = preprocessing.MinMaxScaler()
train_mmx = mm.fit_transform(train_x)
test_mmx = mm.transform(test_x)

#创建Naive Bayes分类器 多项式朴素贝叶斯
mmb = MultinomialNB()
mmb.fit(train_mmx,train_y)
predict_y = mmb.predict(test_mmx)
print ('多项式朴素贝叶斯准确率：{value:.4f}'.format(value=accuracy_score(test_y,predict_y)) ) 

#创建CART决策树
dtc = DecisionTreeClassifier()
dtc.fit(train_mmx,train_y)
predict_y =dtc.predict(test_mmx)
print ('CART决策树准确率：{value:.4f}'.format(value=accuracy_score(test_y,predict_y)) ) 