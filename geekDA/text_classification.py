import os #进行文件处理，读取
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer  #计算单词 TF-IDF 向量的值
from sklearn.naive_bayes import MultinomialNB #朴素贝叶斯分类
from sklearn import metrics #评估预测与实际结果
import jieba 

warnings.filterwarnings('ignore') #警告过滤器
#进行分词
def cut_words(file_path):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    text = open(file_path,'r',encoding='gb18030') .read()#r：以只读方式打开文件。文件的指针将会放在文件的开头。
                                                         #gb18030：中华人民共和国现时最新的内码字集
    text_cut = jieba.cut(text)
    for word in text_cut:
        text_with_spaces += word + ' '
    return text_with_spaces
#获取所以文档分词和标签
def loadfile(file_dir,label):
    """
    将路径下的所有文件加载    
    :param file_dir: 保存txt文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    """
    file_list = os.listdir(file_dir) #返回path指定的文件夹包含的文件或文件夹的名字的列表。
    words_list = []
    labels_list = []
    for file in file_list: #循环文件夹
        file_path =file_dir + '/' + file #获取文件夹下文件
        words_list.append(cut_words(file_path)) #调用cut_words函数
        labels_list.append(label)
    return words_list,labels_list     #Python\geekDA\text classification\test\女性

train_words_list1,train_labels1 = loadfile('./geekDA/text classification/train/女性','女性')
train_words_list2,train_labels2 = loadfile('./geekDA/text classification/train/体育','体育')
train_words_list3,train_labels3 = loadfile('./geekDA/text classification/train/文学','文学')
train_words_list4,train_labels4 = loadfile('./geekDA/text classification/train/校园','校园')
train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

test_words_list1,test_labels1 = loadfile('./geekDA/text classification/test/女性','女性')
test_words_list2,test_labels2 = loadfile('./geekDA/text classification/test/体育','体育')
test_words_list3,test_labels3 = loadfile('./geekDA/text classification/test/文学','文学')
test_words_list4,test_labels4 = loadfile('./geekDA/text classification/test/校园','校园')
test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4
# Python\geekDA\text classification\stop
stop_words = open ('./geekDA/text classification/stop/stopword.txt','r',encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig')# 列表头部\ufeff处理 ；uft-8-sig"中sig全拼为 signature 也就是"带有签名的utf-8
stop_words = stop_words.split('\n') # 根据分隔符分隔
#计算单词权重
tf = TfidfVectorizer(stop_words=stop_words,max_df = 0.5)
train_features = tf.fit_transform(train_words_list)
test_features = tf.transform(test_words_list)
#朴素贝叶斯分类器
clf = MultinomialNB(alpha=0.001).fit(train_features,train_labels)
predict_labels = clf.predict(test_features)

score = metrics.accuracy_score(test_labels,predict_labels)#测试数据 与 预测数据 的准确率
print ('准确率：' , score)
#print('准确率为：', metrics.accuracy_score(test_labels, predicted_labels))
