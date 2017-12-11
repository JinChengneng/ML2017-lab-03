import matplotlib.pyplot as plt #plt 用于显示图片
import matplotlib.image as mpimg #mpimg 用于读取图片
import numpy as np
from scipy import misc  #对图像进行缩放

from feature import NPDFeature

from sklearn.tree import DecisionTreeClassifier

import pickle


#img_face = []
#img_nonface = []

sampleSize = 400
maxIteration = 100

img =[]
img_label = []
img_features = []
img_label_train, img_label_validation = [],[]
img_features_train, img_features_validation = [],[]

def load_img_data():
    # img_face
    for i in range(0,int(sampleSize/2)):
        image = mpimg.imread("./datasets/original/face/face_"+"{:0>3d}".format(i)+".jpg")
        image_gray = rgb2gray(image)
        image_gray_scaled = misc.imresize(image_gray,(24,24))
        img.append(image_gray_scaled) 
        img_label.append(1)
#    #img_nonface
#    for i in range(0,int(sampleSize/2)):
        image = mpimg.imread("./datasets/original/nonface/nonface_"+"{:0>3d}".format(i)+".jpg")
        image_gray = rgb2gray(image)
        image_gray_scaled = misc.imresize(image_gray,(24,24))
        img.append(image_gray_scaled)
        img_label.append(-1)
   

def showImg(item):
    plt.imshow(item)
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def extra_img_features():
    for i in range(0,len(img)):
        f = NPDFeature(img[i])
        features = f.extract()
        img_features.append(features)
        
def get_accuracy(pred, y):
    return sum(pred == y) / float(len(y))

def get_error_rate(pred,y):
    return sum(pred != y) / float(len(y))

if __name__ == "__main__":

    load_img_data()    

#    with open('data', "wb") as f:
#        extra_img_features()
#        pickle.dump(img_features, f)

    
    with open('data', "rb") as f:
        img_features = pickle.load(f)
#        print(img_features)
            
    img_label_train = img_label[0:int(sampleSize*0.7)]
    img_label_validation = img_label[int(sampleSize*0.7):]
    img_features_train = img_features[0:int(sampleSize*0.7)]
    img_features_validation = img_features[int(sampleSize*0.7):]
    
    weights = np.ones(len(img_features_train)) / len(img_features_train)
    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)

#    print(get_accuracy(clf_tree.predict(img_features_train),img_label_train))

    
    pred_train = np.zeros(len(img_features_train))
    
    for i in range(0, maxIteration):
        clf_tree.fit(img_features_train, img_label_train, sample_weight=weights)
        hypothesis = clf_tree.predict(img_features_train)
         # Indicator function
        miss = [int(x) for x in (hypothesis != img_label_train)]
#        print(miss)
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        err_m = np.dot(weights,miss) / sum(weights)
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        weights = np.multiply(weights, np.exp([float(x) * alpha_m for x in miss2]))
         # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [x * alpha_m for x in hypothesis])]
        print((pred_train[13],img_label_train[13]))
        pred_train = np.sign(pred_train)
#        print(sum(pred_train == 1))
        
#        print(get_accuracy(pred_train,img_label_train))
    


