import pandas as pd

train_data = pd.read_csv(r'E:\code_git\Python\GeekDA\train.csv')
test_data = pd.read_csv(r'E:\code_git\Python\GeekDA\test.csv')

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



