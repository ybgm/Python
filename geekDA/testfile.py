import os
import jieba
""" labels = []
labels_map = {'体育':0,'女性':1,'文学':2,'校园':3}
base_path =r'E:\code_git\Python\GeekDA\text classification' 
for root,dirs,files in os.walk (base_path):
    for file in files:
            label = root.split('/')[-1]
            labels.append(label)
print (labels)
 """



""" with open (r'.\text classification\stop\stopword.txt','r',encoding='utf-8') as f:
    STOP_WORDS = [ line.strip() for line in f.readlines() ] #列表生成式；line.strip():去掉每行头尾空白； 依次读取每行

print (STOP_WORDS)  """


""" base_path =r'E:\code_git\Python\GeekDA\text classification'
for root,files in os.walk (base_path):
    print (files)
    #print (dirs) """


def load_data(base_path):
    """ 
    :param base_path:基础路径
    :return：返回分词列表，标签列表
    """
    documents = []
    labels = []
    for root,dirs,files in os.walk ('./text classification/train/'): # os.walk :在目录树中游走输出目录中的文件名，向上或向下；返回root,dirs,files
        print (dirs)
        for file in files:
            label = root.split('\\')[-1] #windows的路径符号自动转成了 \ ，需要转义; [-1]:文件名在最后一个
            labels.append(label)
            filename = os.path.join(root,file)
            with open (filename,'rb') as f: #因为字符集问题，直接使用 二进制rb 读取
                content = f.read()
                word_list = list(jieba.cut(content)) #分词
                words = [wl for wl in word_list] #循环每个词
                documents.append(' '.join(words)) #插入空格
        print (words)             
    return documents,labels

  
""" train_documents,train_labels = load_data('./Python/geekDA/text classification/train')
test_documents,test_labels = load_data('./Python/geekDA/text classification/test')
print (train_documents) """