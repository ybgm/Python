#### 数据分析实战45讲-极客时间

19讲决策树 泰坦尼克专项练习

from sklearn.model_selection import train_test_split #将数据分为测试集和训练集
from sklearn.metrics import accuracy_score #评价指标函数
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.datasets import load_boston #导入sklearn标准数据集-波斯顿房价
from sklearn.tree import DecisionTreeRegressor #导入决策树，CART回归树
from sklearn.feature_extraction import DictVectorizer #特征提取
from sklearn.model_selection import cross_val_score 
import graphviz
from sklearn import tree 


DictVectorizer(sparse = False)

fit_transform
transform
fit
predict
clf.score   clf = DecisionTreeClassifier(criterion= 'entropy')
dot_data = tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(dot_data)
graph.view()

19讲 在模型评估中，决策树提供了 score 函数可以直接得到准确率，但是我们并不知道真实的预测结果，所以无法用预测值和真实的预测结果做比较。我们只能使用训练集中的数据进行模型评估，可以使用决策树自带的 score 函数计算下得到的结果：

sklearn 的全称叫 Scikit-learn，它给我们提供了 3 个朴素贝叶斯分类算法，分别是高斯朴素贝叶斯（GaussianNB）、多项式朴素贝叶斯（MultinomialNB）和伯努利朴素贝叶斯（BernoulliNB）。
高斯朴素贝叶斯：特征变量是连续变量，符合高斯分布，比如说人的身高，物体的长度。
多项式朴素贝叶斯：特征变量是离散变量，符合多项分布，在文档分类中特征变量体现在一个单词出现的次数，或者是单词的 TF-IDF 值等。
伯努利朴素贝叶斯：特征变量是布尔变量，符合 0/1 分布，在文档分类中特征是单词是否出现。

TF-IDF 是一个统计方法，用来评估某个词语对于一个文件集或文档库中的其中一份文件的重要程度。
Term Frequency 和 Inverse Document Frequency
词频 TF
逆向文档频率 IDF
TF-IDF=TF*IDF
什么是停用词？停用词就是在分类中没有用的词，这些词一般词频 TF 高，但是 IDF 很低，起不到分类的作用。

sklearn  TfidfVectorizer 类
TfidfVectorizer(stop_words=stop_words, token_pattern=token_pattern)
tfidf_vec.get_feature_names()
tfidf_vec.vocabulary_)
 tfidf_matrix.toarray()

 from sklearn.naive_bayes import MultinomialNB 
 score
 accuracy_score