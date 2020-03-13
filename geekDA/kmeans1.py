import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from sklearn import preprocessing

def load_data(filePath):
    f = open (filePath,'rb') #只读二进制
    data = []
    #图像像素值
    img = image.open(f)
    #图像尺寸
    width,height = img.size
    for x in range(width):
        for y in range (height):
            #得到点（x,y）的三个通道
            c1, c2, c3 = img.getpixel((x, y))
            data.append([c1,c2,c3])
    f.close()
    #采用Min-Max规范化
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return np.mat(data),width,height

#加载图像，得到规范化的结果img，以及图像尺寸
img,width,height = load_data(r'.\Python\geekDA\weixin.jpg')

#用K-Mean对图像进行2聚类
kmeans = KMeans(n_clusters=2)
kmeans.fit(img)
label = kmeans.predict(img)
#将图像聚类的结果，转化成图像尺寸的矩阵
label = label.reshape([width,height])
#创建新图像，用来保存图像聚类的结果，并设置不同的灰度值
pic_mark = image.new('L',(width,height))
for x in range (width):
    for y in range(height):
        #根据类别设置图像的灰度，类别0，灰度值为255，类别1，灰度值为127
        pic_mark.putpixel((x,y),int(256/(label[x][y]+1))-1)
pic_mark.save(r'.\Python\geekDA\weixin_mark.jpg',"JPEG")