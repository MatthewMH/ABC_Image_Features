# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 05:11:35 2023

@author: CR
"""
# Data Processing
import numpy as np
import glob
import os
import pandas as pd
# Data split
from sklearn.model_selection import train_test_split
# Data Normalization
from sklearn.preprocessing import StandardScaler

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Image Processing
import cv2
# Image viewer and saver
from skimage import io
# Image filter
# gaussian filter nd.gaussian_filter
from scipy import ndimage as nd
# skewness
from scipy.stats import skew
# sobel filter
from skimage.filters import sobel
from skimage.filters import gabor_kernel
# feature extraction
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.measure import label, regionprops, regionprops_table

# Modeling
# K-means Segmentation
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Utils
# For counting the number of images for each category
from collections import Counter
import random
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")



SIZE = 128

train_images = []
train_labels = []

train_images_colored = []

# untuk direktori path di glob glob ("./dataset/train/*)
for directory_path in tqdm(glob.glob("D:/ABCPython-master/Data_daun_melon/data_70_30/train/*")):
    label = directory_path.split('\\')[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_colored = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img_colored = cv2.resize(img_colored, (SIZE, SIZE))
        train_images.append(img)
        train_images_colored.append(img_colored)
        train_labels.append(label)


#lakukan code yang sama untuk data uji
test_images= []
test_labels=[]

test_images_colored = []
# untuk direktori path di glob glob ("./dataset/validation/*")
for directory_path in tqdm(glob.glob("D:/ABCPython-master/Data_daun_melon/data_70_30/validation/*")):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_colored = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img_colored = cv2.resize(img_colored, (SIZE, SIZE))
        test_images.append(img)
        test_images_colored.append(img_colored)
        test_labels.append(fruit_label)


train_images= np.array(train_images, dtype=np.uint8)
train_images_colored = np.array(train_images_colored, dtype=np.uint8)
train_labels= np.array(train_labels)

test_images = np.array(test_images, dtype=np.uint8)
test_images_colored = np.array(test_images_colored, dtype=np.uint8)
test_labels = np.array(test_labels)

from sklearn.preprocessing import LabelEncoder
np.unique(train_labels)

le = LabelEncoder()
le.fit(train_labels)
train_label_encode = pd.DataFrame(le.transform(train_labels))
le.fit(train_labels)
test_label_encode= pd.DataFrame(le.transform(test_labels))

train_label_encode.to_excel("train_label.xlsx")
test_label_encode.to_excel("test_label.xlsx")

np.unique(train_label_encode)

x_train, y_train, x_test, y_test = train_images, train_label_encode, test_images, test_label_encode
x_train.dtype

x_train[0, :, 4].shape

len(train_images_colored[0])
train_images_colored.shape


def feature_extractor(dataset, dataset_colored):
    image_dataset = pd.DataFrame()
    
    for image in tqdm(range(dataset.shape[0])):#iterasi untuk masing masing file
        # data frame sementara untuk mengambil nilai gambar
        # akah dihapus setelah iterasi selesai
        df = pd.DataFrame()
        
        # membuat variabel gambar
        img = dataset[image, :, :]
        
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Fitur greycomatrix
        for d in distances:
            for a in angles:
                GLCM = graycomatrix(img, [d], [a])       
                GLCM_Energy = graycoprops(GLCM, 'energy')[0]
                df[f'Energy_d{d}_a{a}'] = GLCM_Energy
                GLCM_corr = graycoprops(GLCM, 'correlation')[0]
                df[f'Corr_d{d}_a{a}'] = GLCM_corr       
                GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
                df[f'Diss_sim_d{d}_a{a}'] = GLCM_diss       
                GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
                df[f'Homogen_d{d}_a{a}'] = GLCM_hom       
                GLCM_contr = graycoprops(GLCM, 'contrast')[0]
                df[f'Contrast_d{d}_a{a}'] = GLCM_contr
           
        # Fitur entropy 
        entropy = shannon_entropy(img)
        df['Entropy'] = entropy
        
#         #Gabor filter
#         frequencies = [0.1, 0.5, 1]
#         theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#         for freq in frequencies:
#             for angle in theta:
#                 kernel = np.real(gabor_kernel(freq, theta=angle))
#                 filtered = nd.convolve(img, kernel, mode='wrap')
#                 gabor_energy = np.sum(filtered**2)
#                 df[f'Gabor_{freq}_{angle}'] = gabor_energy

#         Color histogram features
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        color_features = cv2.calcHist([color_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        color_features = cv2.normalize(color_features, color_features).flatten()
        for i in range(len(color_features)):
            df[f'Color_{i}'] = color_features[i]
            
        # Color BGR features from cv2.imread()
        R = dataset_colored[image:, :, 2]  # red channel
        G = dataset_colored[image:, :, 1]  # green channel
        B = dataset_colored[image:, :, 0]  # blue channel
        
        meanR = np.mean(R)
        df['meanR'] = meanR
        meanG = np.mean(G)
        df['meanG'] = meanG
        meanB = np.mean(B)
        df['meanB'] = meanB

        varianceR = np.var(R)
        df['varianceR'] = varianceR
        varianceG = np.var(G)
        df['varianceG'] = varianceG
        varianceB = np.var(B)
        df['varianceB'] = varianceB
        
        differenceR = 0.0
        differenceG = 0.0
        differenceB = 0.0

        skewnessR = skew(R.flatten())
        df['skewnessR'] = skewnessR
        skewnessG = skew(G.flatten())
        df['skewnessG'] = skewnessG
        skewnessB = skew(B.flatten())
        df['skewnessB'] = skewnessB
        
       
        # Sobel filters
        #sobel_x = sobel(img, axis=0, mode='reflect')
        #sobel_y = sobel(img, axis=1, mode='reflect')
        #sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        #sobel_dir = np.arctan2(sobel_y, sobel_x)
        
        #sobel_mag_mean = sobel_mag.mean()
        #df['Sobel_mag_mean'] = sobel_mag_mean
        #sobel_mag_var = sobel_mag.var()
        #df['Sobel_mag_var'] = sobel_mag_var
        #sobel_mag_max = sobel_mag.max()
        #df['Sobel_mag_max'] = sobel_mag_max
        #sobel_mag_min = sobel_mag.min()
        #df['Sobel_mag_min'] = sobel_mag_min
        #sobel_dir_mean = sobel_dir.mean()
        #df['Sobel_dir_mean'] = sobel_dir_mean
        #sobel_dir_var = sobel_dir.var()
        #df['Sobel_dir_var'] = sobel_dir_var
        #sobel_dir_max = sobel_dir.max()
        #f['Sobel_dir_max'] = sobel_dir_max
        #sobel_dir_min = sobel_dir.min()
        #df['Sobel_dir_min'] = sobel_dir_min
        
        # Edge detection features
        edges = cv2.Canny(img,100,200)
        edge_features = cv2.calcHist([edges], [0], None, [256], [0, 256])
        edge_features = cv2.normalize(edge_features, edge_features).flatten()
        for i in range(len(edge_features)):
            df[f'Edge_{i}'] = edge_features[i]
        
        image_dataset = pd.concat([image_dataset, df], axis=0)

    return image_dataset

image_features_training = feature_extractor(x_train, train_images_colored)
image_features_validation = feature_extractor(x_test, test_images_colored)

#-----RANDOM FOREST--------
rf = RandomForestClassifier()
rf.fit(image_features_training[['skewnessR', 'skewnessB', 'meanR', 'meanG', 'meanB', 'varianceG', 'skewnessG', 'varianceR']], y_train)
accuracy = accuracy_score(rf.predict(image_features_validation[['skewnessR', 'skewnessB', 'meanR', 'meanG', 'meanB', 'varianceG', 'skewnessG', 'varianceR']]), y_validation)

#image_features_training.to_excel("train.xlsx")
#image_features_validation.to_excel("validation.xlsx")

'''
scaler = StandardScaler()
scaler.fit(image_features_training)
normalized_train_data = scaler.transform(image_features_training)
normalized_test_data = scaler.transform(image_features_validation)


import lightgbm as lgb
from scipy.stats import randint as sp_randint
# parameter turning
from sklearn.model_selection import RandomizedSearchCV

d_train = lgb.Dataset(normalized_train_data, label = y_train)

lgbm_parameter = {'learning_rate':0.05, 'boosting_type': 'dart',
                  'objective':'multiclass',
                  'metric': 'multi_logloss',
                  'num_leaves':100,
                  'max_dept':10,
                  'num_class':4}

lgb_model = lgb.train(lgbm_parameter, d_train, num_boost_round=100)



lgb_predict_training = lgb_model.predict(normalized_train_data)
lgb_predict_validation = lgb_model.predict(normalized_test_data)
# Convert continuous-multioutput array to integer labels
lgb_predict_training = np.argmax(lgb_predict_training, axis=1)
lgb_predict_validation = np.argmax(lgb_predict_validation, axis=1)

lgb_training_accuracy = accuracy_score(y_train, lgb_predict_training)
lgb_validation_accuracy = accuracy_score(y_test, lgb_predict_validation)


print("Training Accuracy: " , lgb_training_accuracy)
print("Validation Accuracy: ", lgb_validation_accuracy)


#print confussion matrik
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, lgb_predict_validation)

fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidth=.5, ax=ax)

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(normalized_train_data, y_train)
rf_predict_train = random_forest.predict(normalized_train_data)
rf_predict_validation = random_forest.predict(normalized_test_data)
rf_train_accuracy = accuracy_score(y_train, rf_predict_train)
rf_validation_accuracy = accuracy_score(y_test, rf_predict_validation)

# Accuracy score
print("Training Accuracy: ", rf_train_accuracy)
print("Validation Accuracy: ", rf_validation_accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, rf_predict_validation)
fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidth=.5, ax=ax)


import xgboost as xgb

xgb_model= xgb.XGBClassifier(objective='multi:softmax', num_class=4, n_estimators=1000, learning_rate=0.05)
xgb_model.fit(normalized_train_data, y_train)
xgb_predict_train = xgb_model.predict(normalized_train_data)
xgb_predict_validation = xgb_model.predict(normalized_test_data)
xgb_train_accuracy = accuracy_score(y_train, xgb_predict_train)
xgb_validation_accuracy = accuracy_score(y_test, xgb_predict_validation)

# Accuracy score
print("Training Accuracy: ", xgb_train_accuracy)
print("Validation Accuracy: ", xgb_validation_accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, xgb_predict_validation)
fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidth=.5, ax=ax)
normalized_test_data.shape
import warnings
warnings.filterwarnings('ignore')
n= random.randint(0, x_test.shape[0]-1)#pilih index untuk diload
img = x_test[n]
plt.imshow(img)

#extra fitur dan reshape

input_img =np.expand_dims(img, axis=0)
input_img_feature = feature_extractor(input_img, test_images_colored)
input_img_feature = np.expand_dims(input_img_feature, axis=0)
input_img_for_RF = np.reshape(input_img_feature, (input_img.shape[0], -1))
input_img_for_RF_normalized = scaler.transform(input_img_for_RF)

#prediksi
img_prediction = lgb_model.predict(input_img_for_RF_normalized)
img_prediction = np.argmax(img_prediction, axis=1)
img_prediction = le.inverse_transform([img_prediction])
print ("prediksinya adalah :", img_prediction)
print("nilai asli label :", test_labels[n])


Model_Recap = pd.DataFrame({'Nama Model': ['LGB', 'Random Forest', 'XGB'], 
              'Training Accuracy': [lgb_training_accuracy, rf_train_accuracy, xgb_train_accuracy],
              'Validation Accuracy': [lgb_validation_accuracy, rf_validation_accuracy, xgb_validation_accuracy]})
Model_Recap.sort_values(by='Validation Accuracy', ascending=False)
'''