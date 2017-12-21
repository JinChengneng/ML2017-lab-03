import matplotlib.pyplot as plt #plt 用于显示图片
import matplotlib.image as mpimg #mpimg 用于读取图片
import numpy as np
from scipy import misc  #对图像进行缩放
from feature import NPDFeature
from sklearn.tree import DecisionTreeClassifier
import pickle

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
#%matplotlib inline

sampleSize = 400
maxIteration = 20

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

    hypothesis_train, hypothesis_validation = [], []
    alpha_m = []

    prediction_train = np.zeros(len(img_features_train),dtype=np.int32)
    prediction_validation = np.zeros(len(img_features_validation),dtype=np.int32)
    
    accuracy_train,accuracy_validation = [] ,[]
    
    for i in range(0, maxIteration):
        print("NUmber of decision trees:",i+1)

        clf_tree.fit(img_features_train, img_label_train, sample_weight=weights)
        hypothesis_train.append (clf_tree.predict(img_features_train) )
        hypothesis_validation.append(clf_tree.predict(img_features_validation))
            
        miss = [int(x) for x in ( hypothesis_train[i] != img_label_train )]
        miss2 = [x if x==1 else -1 for x in miss]
        
        distance = abs(prediction_train - img_label_train) + 1
#        print(distance)
        miss3 = miss * distance * miss2 
#        print(miss3)

        err_m = np.dot(weights,miss)
        if(err_m > 0.5):
            break
        alpha_m.append( 0.5 * np.log( (1 - err_m) / float(err_m)) )
        weights = np.multiply(weights, np.exp([float(x) * alpha_m[i] for x in miss2]))
        weights_sum = weights.sum()
        weights = weights / weights_sum
        
        prediction_train = prediction_train + alpha_m[i] * hypothesis_train[i]
        prediction_validation = prediction_validation + alpha_m[i] * hypothesis_validation[i]
        
        accuracy_train.append( get_accuracy(np.sign(prediction_train),img_label_train) )
        accuracy_validation.append( get_accuracy(np.sign(prediction_validation),img_label_validation) )
        print("Train Accuracy:", accuracy_train[-1])
        print("Validation Accuracy:", accuracy_validation[-1])
        if(accuracy_train[-1] == 1):
            break
        
    plt.xlabel("Number of Decision Trees")
    plt.ylabel("Accuracy")
    plt.plot(accuracy_train, label ="train")
    plt.plot(accuracy_validation, label="validation")
    plt.legend(loc="lower right")

    with open('report.txt', "wb") as f:
        print(classification_report(np.sign(prediction_validation),img_label_validation), file = f)
        


    


