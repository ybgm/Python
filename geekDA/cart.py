from sklearn.model_selection import train_test_split #将数据分为测试集和训练集
from sklearn.metrics import accuracy_score #评价指标函数
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.datasets import load_boston #导入sklearn标准数据集-波斯顿房价
from sklearn.tree import DecisionTreeRegressor #导入决策树，CART回归树

boston=load_boston() # 准备数据集
print (boston.feature_names) #预览数据集
features = boston.data # 获取特征集和房价
prices = boston.target
                        # 随机抽取33%的数据作为测试集，其余为训练集
train_features, test_features, train_price, test_price = train_test_split(features, prices, test_size=0.33)

dtr = DecisionTreeRegressor() #创建CART回归树
dtr.fit(train_features,train_price) # 拟合构造CART回归树
predict_price = dtr.predict(test_features) # 预测测试集中的房价

# 测试集的结果评价
print('回归树二乘偏差均值:', mean_squared_error(test_price, predict_price))
print('回归树绝对值偏差均值:', mean_absolute_error(test_price, predict_price))



