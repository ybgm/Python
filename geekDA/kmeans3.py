import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.image as mpimg

def load_data(filePatah):
    f = open(filePatah,'rb')
    data = []
    img = image.open(f)
    width,height = img.size
    for x in range(width):
        for y in range (height):
            c1, c2, c3 = img.getpixel((x,y))
            data.append([(c1+1)/256.0,(c2+1)/256.0,(c3+1)/256.0])
    f.close()
    return np.mat(data),width,height

img,width,height = load_data(r'.\Python\geekDA\weixin.jpg')
kmeans = KMeans(n_clusters=16)
label = kmeans.fit_predict(img)
label = label.reshape([width,height])

#创建新图像，用来保存图像聚类压缩后的结果
img = image.new('RGB',(width,height))

for x in range(width):
    for y in range(height):
        c1 = kmeans.cluster_centers_[label[x,y],0]
        c2 = kmeans.cluster_centers_[label[x,y],1]
        c3 = kmeans.cluster_centers_[label[x,y],2]
        img.putpixel((x,y),(int(c1*256)-1,int(c2*256)-1,int(c3*256)-1))
img.save(r'.\Python\geekDA\weixin_mark_demm.jpg')