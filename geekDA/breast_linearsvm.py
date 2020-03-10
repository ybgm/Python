import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

data = pd.read_csv(r'.\python\geekda\data23.csv')
pd.set_option('display.max_columns',None)

features_mean =list(data.columns[2:12])
features_se = list (data.columns[12:22])
features_worse = list (data.columns[22:32])

data.drop('id',axis =1,inplace=True)
data['diagnosis'] =  data['diagnosis'].map({'M':1,'B':0})
""" 
print (data.info())
print ('-'*30)
print (data.head()) """

sns.countplot(data['diagnosis'],label= 'count')
#plt.show()
corr= data[features_mean].corr()
plt.figure(figsize=(16,16))
sns.heatmap(corr,annot=True)
#plt.show()

#特征选择
features_remain =  data.columns[1:]
""" features_remain2 =  data.columns[1:31]
print (features_remain)
print (features_remain2) """

train,test = train_test_split(data,test_size=0.3)
train_x = train[features_remain]
train_y = train['diagnosis']
test_x = test[features_remain]
test_y =test['diagnosis']

ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)

model = svm.LinearSVC()
model.fit(train_x,train_y)

prediction = model.predict(test_x)
print ('准确率：',metrics.accuracy_score(prediction,test_y))

testdata= model.predict(train_x)
print ('训练准确率：',metrics.accuracy_score(testdata,train_y))
