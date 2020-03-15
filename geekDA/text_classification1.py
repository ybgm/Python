#-*-coding:utf8-*-
import os
import jieba
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import warnings

#warnings.filterwarnings('ignore') #警告过滤器
#labels_map = {'体育':0,'女性':1,'文学':2,'校园':3}
#加载停用词 text classification\stop\stopword.txt E:\code_git\Python\GeekDA\text classification\stop
with open (r'.\geekdA\text classification\stop\stopword.txt','rb') as f:
    STOP_WORDS = [ line.strip() for line in f.readlines() ] #列表生成式；line.strip():去掉每行头尾空白； 依次读取每行

def load_data(base_path):
    """ 
    :param base_path:基础路径
    :return：返回分词列表，标签列表
    """
    documents = []
    labels = []
    for root,dirs,files in os.walk (base_path): # os.walk :在目录树中游走输出目录中的文件名，向上或向下；返回root,dirs,files
        print (dirs)
        for file in files:
            label = root.split('\\')[-1] #windows的路径符号自动转成了 \ ，需要转义; [-1]:文件名在最后一个
            labels.append(label)
            filename = os.path.join(root,file)
            with open (filename,'rb') as f: #因为字符集问题，直接使用 二进制rb 读取
                content = f.read()
                word_list = jieba.cut(content) #分词
                words = [wl for wl in word_list] #循环每个词
                documents.append(' '.join(words)) #插入空格             
    return documents,labels

def train_nb(train_documents,train_labels,test_documents,test_labels):
    """ 
    :param td:训练集数据
    :param tl:训练集标签
    :param testd:测试集数据
    :param testl:测试集标签
    """
    #计算矩阵
    tt = TfidfVectorizer(stop_words = STOP_WORDS,max_df=0.5)
    train_feature = tt.fit_transform(train_documents)
    #训练模型
    clf = MultinomialNB(alpha=0.001).fit(train_feature,train_labels)
    #模型预测
    test_tf = TfidfVectorizer(stop_words = STOP_WORDS,max_df=0.5,vocabulary=tt.vocabulary_) #vocabulary_：词典索引，对应TF-IDF权重矩阵的列
    test_features = test_tf.fit_transform(test_documents)
    predicted_labels = clf.predict(test_features)
    #获取结果
    x = metrics.accuracy_score(test_labels,predicted_labels)
    return x

train_documents,train_labels = load_data('./geekda/text classification/train')
test_documents,test_labels = load_data('./geekda/text classification/test')
x = train_nb(train_documents,train_labels,test_documents,test_labels)
print ('准确率：' , x)

