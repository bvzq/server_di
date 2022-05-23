# CNN com XGBoost Abelhas
"""
pip install -q tfds-nightly tensorflow matplotlib
pip install tensorflow --upgrade
pip install imagehash
pip install sklearn
pip install scikit-image
pip install xgboost
pip install seaborn
pip install jinja2==2.11
pip install keras_metrics
"""


import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from collections import Counter
from itertools import product

import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import preprocessing
import pathlib
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
import keras_metrics

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
from collections import Counter,defaultdict
import seaborn as sns
import os
import time
import imagehash
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
import matplotlib.cm as cm
from sklearn.model_selection import KFold


bee_ds, bee_info = tfds.load('bee_dataset/bee_dataset_150',split='train',shuffle_files=True,with_info=True)

bee_df = tfds.as_dataframe(bee_ds,bee_info)

def convert_to_grayscale(img):
    #Convert to tensor
    tensor_img = tf.convert_to_tensor(img)
    #Convert from rgb to grayscale
    img_bw = tf.image.rgb_to_grayscale(tensor_img)

    return img_bw
    
def hash_image(img):
    img_bw = convert_to_grayscale(img)
    #Hash Image
    #hash_img = str(imagehash.phash(tf.keras.utils.array_to_img(img_bw.numpy())))
    hash_img = str(imagehash.phash(tf.keras.preprocessing.image.array_to_img(img_bw.numpy())))
    return hash_img

#Checking for duplicated images
counts_imgs = dict()
arr_dups = list()
#Loop over images
for i, img in enumerate(bee_df['input'].values):
    #hash image
    hash_img = hash_image(img)
    #Add to dataframe
    bee_df.loc[i,'hash'] = hash_img

"""
get_image: Plots image  
Arguments:
  df = DataFrame where images are stored
  img_index = position of image
  axe = pass axe to plot if subplots are needed
"""
def plot_image(df,img_index,axe=None):
    if axe is not None:
      #axe.imshow(tf.keras.utils.array_to_img(df.loc[img_index,'input']))
      axe.imshow(tf.keras.preprocessing.image.array_to_img(df.loc[img_index,'input']))
    else:
      #plt.imshow(tf.keras.utils.array_to_img(df.loc[img_index,'input']))
      plt.imshow(tf.keras.preprocessing.image.array_to_img(df.loc[img_index,'input']))
      plt.show()

#Loop over array of indexes of duplicated images
dups = bee_df.index[bee_df.duplicated(subset=['hash'],keep=False)]

for i in range(0,len(dups)-1,2):
    #Set axis to plot images
    fig, (ax1, ax2) = plt.subplots(1,2)
    #Set title
    fig.suptitle("Duplicated Images")
    #Plot images
    plot_image(bee_df,dups[i],ax1)
    plot_image(bee_df,dups[i+1],ax2)

bee_df['input'] = bee_df['input'].apply(lambda x: (x / 255).astype(np.float32))

bee_df = bee_df.drop(dups)

idx = bee_df.index[(bee_df['output/wasps_output']==1.0) & (bee_df['output/pollen_output']==1.0) & (bee_df['output/cooling_output']==1.0)]
bee_df = bee_df.drop(idx)

for i, val in enumerate(bee_df.values[:,1:3]):
    if val[0] == 0.0 and val[1] == 0.0:
        bee_df.loc[i,'bee_job'] = 0.0
    elif val[0] == 1.0 and val[1] == 0.0:
        bee_df.loc[i,'bee_job'] = 1.0
    elif val[0] == 0.0 and val[1] == 1.0:
        bee_df.loc[i,'bee_job'] = 2.0
    elif val[0] == 1.0 and val[1] == 1.0:
        bee_df.loc[i,'bee_job'] = 3.0

def convert_rgb_gray(img):
    return rgb2gray(img)

def binarize_image(img,threshold):
      img_binarize = img > threshold
      return img_binarize

def detect_blobs(data):
    blobs = list()
    for i in data.index:
        #Detect blobs
        img_bw = convert_rgb_gray(data[i])
        img_binarize = binarize_image(img_bw,0.6)
        #blob_doh(img_binarize,max_sigma=40,threshold=0.01
        blobs.append({"blobs":blob_doh(img_binarize,min_sigma=20,max_sigma=40,threshold=0.01),
                  "img_index":i})
    return blobs

def plot_blobs(data,arr_blobs):
    for i in range(len(data)):
        img_bw = convert_rgb_gray(data[i])
        img_binarize = binarize_image(img_bw,0.6)
        fig, ax = plt.subplots(1,1,figsize=(10,5))
        plt.imshow(img_binarize,cmap='gray')
    for i, blobs in enumerate(arr_blobs):
        for blob in blobs:
            y,x, area = blob
            ax.add_patch(plt.Circle((x, y), area, color='r', 
                                fill=False))
      #plt.show()

def get_blobs_area(data,blobs_df):
    img_blobs_area = list()
    arr_blobs,index = blobs_df['blobs'].values,blobs_df['img_index'].values
    for i,blobs in enumerate(arr_blobs):
        for blob in blobs:
            #x,y,area = blob
            x = blob[0].astype(np.int64)
            y = blob[1].astype(np.int64)
            area = blob[2]
            r = np.ceil(np.sqrt(area/np.pi)).astype(np.int64)
            #print("x:{},y:{},r:{}".format(x,y,r))
            r = r + np.ceil(2*np.pi*r).astype(np.int64)
            if x > r:
                idx_rows = np.s_[x-r:x+r]
            else:
                idx_rows = np.s_[x:x+r]
      
            if y > r:
                idx_cols = np.s_[y-r:y+r]
            else:
                idx_cols = np.s_[y:y+r]
      
            img_blobs_area.append({"blob_area":data[index[i]][idx_rows,idx_cols],"img_index":index[i]})
    return img_blobs_area

bee_only_df = bee_df.loc[(bee_df['output/wasps_output'] == 0.0) &
                        (bee_df['output/pollen_output'] == 0.0 )&
                        (bee_df['output/cooling_output'] == 0.0) &
                        (bee_df['output/varroa_output'] == 0.0)]
#bee_only_df = bee_only_df.drop(columns=['output/cooling_output','output/pollen_output','output/varroa_output'])
#plt.imshow(bee_only_df['input'].values[0])
print("Number of images:", len(bee_only_df.values))

blobs_df = pd.DataFrame(detect_blobs(bee_only_df['input']))

blobs_area_df = pd.DataFrame(get_blobs_area(bee_only_df['input'],blobs_df))

#resize images
x_resized = list()
for i in range(len(blobs_area_df['blob_area'])):
    #img = tf.convert_to_tensor(np.asarray(blobs_area_df['blob_area'][i]).astype(np.float32) / 255)
    img = convert_to_grayscale(blobs_area_df['blob_area'][i])
    img = tf.image.resize(img,[40,40],method='bilinear').numpy().flatten()
    x_resized.append(img)
  
x_resized = np.asarray(x_resized)

kmeans = KMeans(n_clusters=2,random_state=0)
kmeans.fit(x_resized)

def get_cluster_idx(labels,cluster_label):
    return np.where(labels == cluster_label)

idx_cluster0 = get_cluster_idx(kmeans.labels_,0)[0]
idx_cluster1 = get_cluster_idx(kmeans.labels_,1)[0]

#for i in range(len(kmeans.labels_[:1000])):
#    print("Cluster:{},img_index:{}".format(kmeans.labels_[i],blobs_area_df.loc[i,'img_index']))

idx_imgs_cluster1 = np.unique(blobs_area_df.loc[idx_cluster0,'img_index'].values)
bee_df = bee_df.drop(idx_imgs_cluster1)

X_train, X_test, y_train, y_test = train_test_split(bee_df['input'].values,bee_df[bee_df.columns[1:]].values,test_size=0.33, random_state=0)

bee_df_train = pd.DataFrame({'input':X_train})
bee_df_train['output/cooling_output'] = y_train[:,0]
bee_df_train['output/pollen_output'] = y_train[:,1]
bee_df_train['output/varroa_output'] = y_train[:,2]
bee_df_train['output/wasps_output'] =  y_train[:,3]
bee_df_train['bee_job'] =  y_train[:,5] #y_train[:,4]

bee_df_test = pd.DataFrame({'input':X_test})
bee_df_test['output/cooling_output'] = y_test[:,0]
bee_df_test['output/pollen_output'] = y_test[:,1]
bee_df_test['output/varroa_output'] = y_test[:,2]
bee_df_test['output/wasps_output'] =  y_test[:,3]
bee_df_test['bee_job'] =  y_test[:,5]

def data_augmentation(ds,labels,n):
    if n > len(ds):
        raise Exception("n argument is bigger than sample size")
    ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(ds[:n]),tf.convert_to_tensor(labels[:n])))
    data_augmentation = tf.keras.Sequential([
                              layers.RandomFlip("horizontal_and_vertical"),
                              #layers.RandomRotation(0.1),
                              layers.RandomContrast(0.3),
                              layers.RandomTranslation(0.1,0.1)
                              #layers.Input(shape=(150,75,3)),
                              #layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                              #layers.experimental.preprocessing.RandomContrast(0.3),
                              #layers.experimental.preprocessing.RandomTranslation(0.1,0.1)
                              
    ])

    ds = ds.map(lambda x,y:(data_augmentation(x,training=True),y))
    return ds

def assign_ds_to_df(df,ds):
    temp = list()
    for i, (x,y) in enumerate(ds):
        temp.append({"input":x.numpy(),
            'output/cooling_output':y.numpy()[0],
            'output/pollen_output':y.numpy()[1],
            'output/varroa_output':y.numpy()[2],
            'output/wasps_output':y.numpy()[3]})

    temp_df = pd.DataFrame(temp)
  
    return pd.concat([df,temp_df],ignore_index=True)  
  
def augment_data(df,cond,n):
    ds = np.asarray(df.loc[cond,'input'].tolist()).astype(np.float32)
    labels = np.asarray(df.loc[cond,df.columns[1:]]).astype(np.float32)
    data_aug = data_augmentation(ds,labels,n)
    return assign_ds_to_df(df,data_aug)

bee_df_train = augment_data(bee_df_train,(bee_df_train['output/wasps_output'] == 1.0),600)
bee_df_train = augment_data(bee_df_train,(bee_df_train['output/wasps_output'] == 1.0),600)

bee_df_train = augment_data(bee_df_train,(bee_df_train['output/varroa_output'] == 1.0),800)
bee_df_train = augment_data(bee_df_train,(bee_df_train['output/varroa_output'] == 1.0),800)

for i in range(0,2):
    bee_df_train = augment_data(bee_df_train,(bee_df_train['output/cooling_output'] == 1.0),700)
    bee_df_train = augment_data(bee_df_train,(bee_df_train['output/cooling_output'] == 1.0),700)
    bee_df_train = augment_data(bee_df_train,(bee_df_train['output/pollen_output'] == 1.0),600)
    bee_df_train = augment_data(bee_df_train,(bee_df_train['output/pollen_output'] == 1.0),600)
    bee_df_train = augment_data(bee_df_train,(bee_df_train['output/pollen_output'] == 1.0 )&(bee_df['output/cooling_output'] == 1.0),90)

def create_ds(train,cond_train,target,isSeries=True):
    X_train = np.asarray(list(train.loc[cond_train,'input'])).astype(np.float32)
  #X_test = np.asarray(list(test.loc[cond_test,'input'])).astype(np.float32)
  
    if isSeries:
        y_train = np.asarray(list(train.loc[cond_train,target])).astype(np.float32)
    #y_test = np.asarray(list(test.loc[cond_test,target])).astype(np.float32)
    else:
        y_train = np.asarray(list(train.loc[cond_train,target].values)).astype(np.float32)
    #y_test = np.asarray(list(test.loc[cond_test,target].values)).astype(np.float32)

  #return train_ds,test_ds

    return X_train,y_train


cond_train = (bee_df_train['output/wasps_output'] == 0.0) | (bee_df_train['output/wasps_output'] == 1.0)
X_train_bee,y_train_bee = create_ds(bee_df_train,cond_train,'output/wasps_output')

cond = (bee_df_train['output/varroa_output'] == 0.0) | (bee_df_train['output/varroa_output'] == 1.0)
X_train_varroa, y_train_varroa = create_ds(bee_df_train,cond,'output/varroa_output')

cond = (bee_df_train['output/cooling_output'] == 0.0) | (bee_df_train['output/cooling_output'] == 1.0) | (bee_df_train['output/pollen_output'] == 0.0) | (bee_df_train['output/pollen_output'] == 1.0)
X_train_multi,y_train_multi= create_ds(bee_df_train,cond,bee_df_train.columns[1:3],False)


y_train_bee_job_XGB = np.asarray(list(bee_df_train.loc[(bee_df_train['output/cooling_output'] == 0.0) | (bee_df_train['output/cooling_output'] == 1.0) | (bee_df_train['output/pollen_output'] == 0.0) | (bee_df_train['output/pollen_output'] == 1.0),'bee_job'])).astype(np.float32)
y_test_bee_job_XGB = np.asarray(list(bee_df_test.loc[(bee_df_test['output/cooling_output'] == 0.0) | (bee_df_test['output/cooling_output'] == 1.0) | (bee_df_test['output/pollen_output'] == 0.0) | (bee_df_test['output/pollen_output'] == 1.0),'bee_job'])).astype(np.float32)

def create_model(instructions,dims):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=dims))
    for instruct in instructions:
        for key in instruct:
            if key == 'Conv2D':
                model.add(layers.Conv2D(instruct[key][0],instruct[key][1],padding='same',activation='relu'))
            elif key == 'BatchNormalization':
                model.add(layers.BatchNormalization(momentum =instruct[key][0],epsilon=instruct[key][1]))
            elif key == 'MaxPooling2D':
                model.add(layers.MaxPooling2D(instruct[key][0]))
            elif key == 'Dropout':
                model.add(layers.Dropout(instruct[key][0]))
            elif key == 'Dense':
                model.add(layers.Dense(instruct[key][0],activation=instruct[key][1]))
            elif key == 'Flatten':
                model.add(layers.Flatten(name=instruct[key][0]))
            else:
                raise Exception("Provide one the following layer types: [Conv2D, BatchNormalization, MaxPooling, Dropout, Dense, Flatten]")
    return model

params = {'filters__Conv2D':[[16,32,64,128],[8,16,32,64]],
          'kernel_size__Conv2D': [2,3],
          'momentum__BatchNormalization':[0.2],
          'epsilon__BatchNormalization':[1e-4],
          'pool_size__MaxPooling2D':[2],
          'rate__Dropout':[0.2,0.4],
          'units__Dense': [1],
          'activation__Dense': ['relu'],
          'name__Flatten':['flatten_layer']}
dims = (150,75,3)

def grid_search(params):
    combinations = list(product(*(params[key] for key in params)))
    models = list()
    for comb in combinations:
        model_def = [
                    {'Conv2D':[comb[0][0],comb[1]]},
                    {'BatchNormalization':[comb[2],comb[3]]},
                    {'Conv2D':[comb[0][1],comb[1]]},
                    {'BatchNormalization':[comb[2],comb[3]]},
                    {'MaxPooling2D':[comb[4]]},
                    {'Conv2D':[comb[0][2],comb[1]]},
                    {'Conv2D':[comb[0][2],comb[1]]},
                    {'Conv2D':[comb[0][2],comb[1]]},
                    {'MaxPooling2D':[comb[4]]},
                    {'Conv2D':[comb[0][2],comb[1]]},
                    {'Conv2D':[comb[0][2],comb[1]]},
                    {'Conv2D':[comb[0][2],comb[1]]},
                    {'MaxPooling2D':[comb[4]]},
                    {'Conv2D':[comb[0][3],comb[1]]},
                    {'Conv2D':[comb[0][3],comb[1]]},
                    {'Conv2D':[comb[0][3],comb[1]]},
                    {'MaxPooling2D':[comb[4]]},
                    {'Dropout':[comb[5]]},
                    {'Flatten':[comb[8]]},
                    {'Dense':[comb[6],comb[7]]}
                 ]
        models.append(model_def)
    return models
  #return model with combination of params

tf.config.experimental_run_functions_eagerly(True)

def cv(X,y,params,cv_splits,dims=None,isNN=True): # loss = tf.keras.losses.BinaryCrossentropy()
    #***CNN****
    if isNN:
        print("isNN")
        kf = KFold(n_splits=cv_splits,shuffle=True,random_state=0)
        history_models = list()
        models_def = grid_search(params)
        for i, model_def in enumerate(models_def):
      #Clear model from memory
            tf.keras.backend.clear_session()
      #Create Model
            model = create_model(model_def,dims)
      #Name Model
            model._name = 'cnn_{}'.format(i)
      #Compile Model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])
      #Save initial weights to reset after each fold
            model.save_weights('initial_weights.h5')
            print("i ", i, "model_def", model_def)
            print("summary", model.summary())
            i=0
            for train_idx,test_idx in kf.split(X):
                print("Fold number:",i)
            #Reset Weights
                model.load_weights('initial_weights.h5')
        #get train-test set
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                batch_values = np.arange(50,150,25)
        #for batch in batch_values:
        #Fit model
                epochs = 15
                history = model.fit(
                    x=X_train,
                    y=y_train,
                    validation_data=(X_test,y_test),
                    epochs=epochs,
                    batch_size = 90)
                i+=1
                history_models.append({'model_name':model.name,'model_def':model_def,'history':history})
            return history_models
  #***XGBOOST******
    else:
        print("isXGB")
        xgb = XGBClassifier(eval_metric='error')
        gcv = GridSearchCV(xgb,params,cv=cv_splits)
        gcv.fit(X,y)
        return gcv.cv_results_
        

def get_scores(history,n_splits):
    hist_df = pd.DataFrame(history)
    scores = list()
    for i in range(n_splits,len(histories)+n_splits,n_splits):
        vals_accuracy_scores = list()
        vals_loss_scores = list()
        for j in range(i-n_splits,i):
            #Append validation accuracy from last epoch
            vals_accuracy_scores.append(hist_df['history'].values[j].history['val_accuracy'][-1])
            vals_loss_scores.append(hist_df['history'].values[j].history['val_loss'][-1])
        #get model name
        model_name = str(hist_df.loc[i-n_splits,'model_name'])
        #Append metrics according to the performance of each model
        scores.append({"model_name":model_name,
         'val_accuracy_mean':np.mean(vals_accuracy_scores),
         'val_loss_mean':np.mean(vals_loss_scores)})
    return scores



#### varroa
print("varroa")
n_splits = 4
histories = cv(X_train_varroa,y_train_varroa,params,n_splits,dims)

scores_df = pd.DataFrame(get_scores(histories,n_splits))
print(scores_df)
scores_df.to_csv(r'C:\Users\bvazq\Desktop\documentos\universidade\2semestre\AAA\projeto\servidorDI\CNN_varroa.txt')

##### bee wasp
#n_splits = 4
#histories = cv(X_train_bee,y_train_bee,params,n_splits,dims)

#scores_df = pd.DataFrame(get_scores(histories,n_splits))
#print(scores_df)
#scores_df.to_csv(r'C:\Users\bvazq\Desktop\documentos\universidade\2semestre\AAA\projeto\servidorDI\CNN_bee.txt')



#### job
print("multi job")
n_splits = 4
histories = cv(X_train_multi,y_train_multi,params,n_splits,dims) # , tf.keras.losses.CategoricalCrossentropy()

scores_df = pd.DataFrame(get_scores(histories,n_splits))
print(scores_df)
scores_df.to_csv(r'C:\Users\bvazq\Desktop\documentos\universidade\2semestre\AAA\projeto\servidorDI\CNN_multi.txt')







