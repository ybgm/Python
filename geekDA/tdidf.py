from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer()
documents = [ 'this is the bayes document', 
            'this is the second second document', 
             'and the third one', 
            'is this the document']

tfidf_matrix = tfidf_vec.fit_transform(documents)

print('不重复的词:', tfidf_vec.get_feature_names())
print('每个单词的ID:', tfidf_vec.vocabulary_)
print('每个单词的tfidf值:', tfidf_matrix.toarray())




import nltk  #用于英文文档
word_list = nltk.word_tokenize(text) #分词
nltk.pos_tag(word_list) #标注单词的词性

import jieba #用于中文文档
word_list = jieba.cut (text) #中文分词
#加载停用词表
stop_words = [line.strip().decode('utf-8') for line in io.open('stop_words.txt').readlines()]
#计算单词的权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)  #max_df:最高出现频率
features = tf.fit_transform(train_contents)

# 多项式贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB  
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
#测试集的特征矩阵
test_tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5, vocabulary=train_vocabulary)
test_features=test_tf.fit_transform(test_contents)
#使用生成的分类器做预测
predicted_labels=clf.predict(test_features)
#计算准确率
from sklearn import metrics
print (metrics.accuracy_score(test_labels, predicted_labels)