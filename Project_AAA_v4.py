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
pip install DNN
"""


#!/usr/bin/env python
# coding: utf-8

# ### Coisas a fazer:
# - Fazer undersampling básico para as classificações binárias
# 
# - Undersampling + Oversampling (zoom, virar, deslocar...) Para o modelo multi classes (os 93) podemos fazer até x10
# 
# - Dividir a imagem em Bins para analisar onde de acumula o hashing, usar como método de semelhança (isto não o percebo muito bem)
# 
# - Falamos nos blobs, mas eu interpreto que não é necessário (só se tivermos tempo)
# 
# - Usar padding mínimo se fazemos CNN com imagens rectangulares
# 
# - Modelo com autoencoder pode ser menos interessante
# 
# - Começar com as imagens menos resolutivas e redes mais simples, depois vamos incrementando (**Atual: 150x75**)
# 
# - Validação-test 90:10 ou 80:20 são ambas boas proporções
# 
# - Modelo preto/branco mais rápido
# 
# - Fazer atenção ao Pacience (5epocas), quando a loss desce e a accuracy fica estável é sintoma de overfitting, podemos usar esta situação para parametrizar a Pacience
# 
# - Número de parâmetros ao final da rede deve servir proporcional ao tamanho da imagem, não deve ter excesso de parâmetros
# 
# - A accuracy só para a validação final, entre epochs usamos a loss 
# 
# - O batch normalization standariza os delta intra-camada
# 
# - Pooling Max
# 
# - Remover imagens Erradas
# 
# - Lidar com as imagens duplicadas
# 
# - (Gerar imagens) (Isto vai para anexo código no relatório)

# In[ ]:


#!pip install -q tfds-nightly tensorflow matplotlib


# In[ ]:


#!pip install tensorflow --upgrade


# In[ ]:


#!pip install imagehash


# In[ ]:


#!pip install scikit-image


# In[ ]:


#!pip install xgboost


# In[6]:


#!pip install --upgrade jinja2


# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


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
from sklearn.metrics import silhouette_score,silhouette_samples,confusion_matrix,recall_score,precision_score
import matplotlib.cm as cm
from sklearn.model_selection import KFold
from skimage.transform import resize

# # 1. **Load Data**

# In[3]:


bee_ds, bee_info = tfds.load('bee_dataset/bee_dataset_150',split='train',shuffle_files=True,with_info=True)


# In[4]:


#tfds.show_examples(bee_ds,bee_info)


# # 2. **Exploratory Data Analysis**

# Before starting the EDA task, the bee dataset will be converted to a dataframe in order to ease the EDA phase. Alongi

# In[4]:


bee_df = tfds.as_dataframe(bee_ds,bee_info)


# In[6]:


bee_df.columns


# Our dataset is composed by the input or X that corresponds to the images and we have 4 target binary variables.

# In[22]:


print("Size of a image:",bee_df['input'][0].shape)


# The images have a size of 300x150 and as expected are on rgb.

# In[23]:


#Checking for Missing values
bee_df.isna().value_counts()


# In[5]:


def convert_to_grayscale(img):
  #Convert to tensor
  tensor_img = tf.convert_to_tensor(img)
  #Convert from rgb to grayscale
  img_bw = tf.image.rgb_to_grayscale(tensor_img)

  return img_bw


# In[6]:


def hash_image(img):
  #img_bw = convert_to_grayscale(img)
  #Hash Image
  #hash_img = str(imagehash.phash(tf.keras.utils.array_to_img(img_bw.numpy())))
  hash_img = str(imagehash.phash(tf.keras.preprocessing.image.array_to_img(img_bw.numpy())))
  return hash_img


# Check Duplicates

# In[7]:


#Checking for duplicated images
counts_imgs = dict()
arr_dups = list()
#Loop over images
for i, img in enumerate(bee_df['input'].values):
  #hash image
  hash_img = hash_image(img)
  #Add to dataframe
  bee_df.loc[i,'hash'] = hash_img


# In[8]:


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


# In[9]:


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
  


# In[10]:


"""
plot_class_counts: Plots a barplot with the respective number of instances existed on the dataset for a given target variable
Arguments:
  df = DataFrame to perform the counting (pd.DataFrame)
  target = target variable/class to count the number of examples (string)
  title = title of the plot (string)
  labels = labels to be plotted on the x axis (list of strings)
"""
def plot_class_counts(df,target,title,labels):
    fig,ax = plt.subplots(figsize=(6,6))
    #vals = df.value_counts(subset=[target])
    blist = plt.bar(labels,df.value_counts(subset=[target]),color=['#30A5BF','#F2BE22'])
    #blist[1].set_color('y')
    plt.title(title)
    total = len(df.values)
    for p in ax.patches:
      #Get percentages for classes
          percentage = '{:.1f}%'.format(100 * (p.get_height()/total))
          x = p.get_x() + p.get_width()/2 - 0.05
          y = p.get_y() + p.get_height() + 50
          ax.annotate(percentage, (x, y))
      #Get absolute values for classes
          x = p.get_x() + p.get_width()/2 - 0.05
          y = p.get_y() + p.get_height()/2
          ax.annotate(p.get_height(),(x,y))


# In[192]:


#plot_class_counts(bee_df,'output/wasps_output','Number of images of bees and wasps',['bees','wasps'])


# Analysing the graphic, clearly there is an imbalanced of classes, where 87.3% of dataset is composed by bees' images and only 12.7% of the images correspond to wasps.

# In[31]:


#plot_class_counts(bee_df,'output/pollen_output','Number of images of bees with and without polen',['polen_no','polen_yes'])


# Similarly, to the previous situations, there is another imbalance of classes regarding the images that polen is present or not. Existing only 1035 images where this is true.  

# In[32]:


#plot_class_counts(bee_df,'output/cooling_output','Number of images of bees performing cooling',['cooling_no','cooling_yes'])


# For the cooling target variable, it exists another imbalance of classes, where 15.2% of images corresponds to bees performing the cooling process.

# In[33]:


#plot_class_counts(bee_df,'output/varroa_output','Number of images of bees sick with varroa and healthy',['varroa_no','varroa_yes'])


# Again, for the varroa, sickness, target variable, it is verified another imbalance of classes, where 16.2% of images corresponds to bees sick with the varroa.

# In[34]:


bee_info_df = bee_df[['output/cooling_output','output/pollen_output','output/varroa_output','output/wasps_output']]
bee_info_df['sum'] = bee_df[['output/cooling_output','output/pollen_output','output/varroa_output','output/wasps_output']].sum(axis = 1)
print(len(bee_info_df[bee_info_df['sum'] >1]), "Observations with more than a caracteristic","\n")

# na tabela pivot table vemos um resumo das possíveis combinaçoes entre variáveis dependentes.
print(pd.crosstab(bee_info_df['output/cooling_output'],bee_info_df['output/pollen_output']),"\n")
print(pd.crosstab(bee_info_df['output/cooling_output'],bee_info_df['output/varroa_output']),"\n")
print(pd.crosstab(bee_info_df['output/pollen_output'],bee_info_df['output/varroa_output']),"\n")
print(pd.crosstab(bee_info_df['output/pollen_output'],bee_info_df['output/wasps_output']),"\n")
print(pd.crosstab(bee_info_df['output/wasps_output'],bee_info_df['output/varroa_output']),"\n")
print(pd.crosstab(bee_info_df['output/cooling_output'],bee_info_df['output/wasps_output']),"\n")


# From analysing the possibles situations where we might have wasps with polen, or cooling. It was found that this happend for one case of each situation. Furthermore, it was verified that there are 93 cases where cooling and polen target variables were true.
# 
# Therefore, we consider this outliers each demand our attention to handle them carefully.

# In[35]:


"""
get_plot_outliers: Selects and plots the images of the outliers identified
Arguments:
  df = DataFrame (pandas.DataFrame)
  target1 = target variable/class (string)
  target2 = target variable/class (string)
"""
def get_plot_outliers(df,target1,target2):
  #Get Index or array of indexes where condition is true
  idx = bee_df.index[(bee_df[target1]==1.0) & (bee_df[target2]==1.0)]
  #Loop over the indexes
  for i in idx:
    #Plot image
    plt.imshow(tf.keras.utils.array_to_img(bee_df.loc[i,'input']))
    plt.show()


# IMAGEM ERRADA

# In[36]:


#Get example where wasp is true and polen is true
#get_plot_outliers(bee_df,'output/wasps_output','output/pollen_output')


# IMAGEM ERRADA

# In[37]:


#Get example where wasp is true and cooling is true
#get_plot_outliers(bee_df,'output/wasps_output','output/cooling_output')


# In[38]:


#Get example where pollen is true and cooling is true
#get_plot_outliers(bee_df,'output/pollen_output','output/cooling_output')


# In[39]:


#Get example where varroa is true and cooling is true
 #get_plot_outliers(bee_df,'output/varroa_output','output/cooling_output')


# # 3.1 **Pre-Processing Data**
# 
# 
# 

# ### Fix The imbalance dataset

# Normalization of images

# In[11]:


bee_df['input'] = bee_df['input'].apply(lambda x: (x / 255).astype(np.float32))


# Remove Duplicates

# In[12]:


bee_df = bee_df.drop(dups)


# Remover Imagem errada

# In[13]:


idx = bee_df.index[(bee_df['output/wasps_output']==1.0) & (bee_df['output/pollen_output']==1.0) & (bee_df['output/cooling_output']==1.0)]
bee_df = bee_df.drop(idx)


# Blobs

# In[14]:


def convert_rgb_gray(img):
  return rgb2gray(img)


# In[15]:


def binarize_image(img,threshold):
  img_binarize = img > threshold
  return img_binarize


# In[16]:


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


# In[17]:


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
      plt.show()


# X,y são as coordenas centrais do blob
# Através da área conseguimos encontrar o raio
# Através da soma do raio às coordenas conseguimos selecionar os pixels dentro do blob
# 
# ter em atenção situações exceção quando a subtração de x-r ou y-r é um número negativo

# In[18]:


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


# In[19]:


def plot_image_blob_area(data,blobs_area):
  for i in range(len(data)):
    for area in blobs_area:
      plt.imshow(area)
      plt.show()


# In[20]:


bee_only_df = bee_df.loc[(bee_df['output/wasps_output'] == 0.0) &
                        (bee_df['output/pollen_output'] == 0.0 )&
                        (bee_df['output/cooling_output'] == 0.0) &
                        (bee_df['output/varroa_output'] == 0.0)]
#bee_only_df = bee_only_df.drop(columns=['output/cooling_output','output/pollen_output','output/varroa_output'])
#plt.imshow(bee_only_df['input'].values[0])
#print("Number of images:", len(bee_only_df.values))


# In[21]:


blobs_df = pd.DataFrame(detect_blobs(bee_only_df['input']))


# In[52]:


#plot_blobs(bee_only_df['input'].head(10).values,blobs_df['blobs'])


# In[22]:


blobs_area_df = pd.DataFrame(get_blobs_area(bee_only_df['input'],blobs_df))
blobs_area_df


# In[54]:


run = False
if run:
    plot_image_blob_area(bee_only_df['input'].head(10),blobs_area_df['blob_area'])


# Clustering blobs

# In[23]:

#resize images
x_resized = list()
for i in range(len(blobs_area_df['blob_area'])):
    #img = tf.convert_to_tensor(np.asarray(blobs_area_df['blob_area'][i]).astype(np.float32) / 255)
    img = convert_rgb_gray(blobs_area_df['blob_area'][i])
    #img = tf.image.resize(img,[40,40],method='bilinear').numpy().flatten()
    img = resize(img,output_shape=[40,40]).flatten()
    x_resized.append(img)
# In[24]:


x_resized = np.asarray(x_resized)
#x_resized = np.squeeze(np.asarray(x_resized))

# In[25]:


kmeans = KMeans(n_clusters=2,random_state=0)
kmeans.fit(x_resized)


# In[26]:


def get_cluster_idx(labels,cluster_label):
  return np.where(labels == cluster_label)


# In[27]:


def plot_examples_on_cluster(idx_cluster,cluster_label,n):
  print("Cluster {}".format(cluster_label))
  for i,val in enumerate(idx_cluster[:n]):
    plt.imshow(blobs_area_df['blob_area'][val])
    plt.show()


# In[28]:


idx_cluster0 = get_cluster_idx(kmeans.labels_,0)[0]
idx_cluster1 = get_cluster_idx(kmeans.labels_,1)[0]


# In[61]:


#for i in range(len(kmeans.labels_[:1000])-5):
  #print(kmeans.labels_[i:i+5])

#for i in range(len(kmeans.labels_[:1000])):
  #print("Cluster:{},img_index:{}".format(kmeans.labels_[i],blobs_area_df.loc[i,'img_index']))


# ### Cluster 0

# In[62]:


#plot_examples_on_cluster(idx_cluster0,0,100)


# ### Cluster 1

# In[63]:


#plot_examples_on_cluster(idx_cluster1,1,100)


# Eliminar Imagens do Cluster 0

# In[29]:


idx_imgs_cluster1 = np.unique(blobs_area_df.loc[idx_cluster0,'img_index'].values)
#Delete images from cluster 1
bee_df = bee_df.drop(idx_imgs_cluster1)


# In[30]:


bee_df.value_counts(subset=['output/wasps_output'])


# In[31]:


print(len(bee_df.loc[(bee_df['output/wasps_output']==0) & (bee_df['output/pollen_output'] == 0.0 )&
                        (bee_df['output/cooling_output'] == 0.0) &
                        (bee_df['output/varroa_output'] == 0.0)].values))


# In[32]:


#plot_class_counts(bee_df,'output/wasps_output','Number of images of bees and wasps',['bees','wasps'])


# In[33]:


#plot_class_counts(bee_df,'output/pollen_output','Number of images of bees with and without polen',['polen_no','polen_yes'])


# In[31]:


augment_bee_df = bee_df.copy().drop(columns=['hash']).reset_index(drop=True)


# ### Data Augmentation

# In[32]:


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


# In[33]:


def assign_ds_to_df(df,ds):
    temp = list()
    for i, (x,y) in enumerate(ds):
        temp.append({"input":x.numpy(),
                'output/cooling_output':y.numpy()[0],
                'output/pollen_output':y.numpy()[1],
                'output/varroa_output':y.numpy()[2],
                'output/wasps_output':y.numpy()[3]}) #,'bee_job':y.numpy()[4]

    temp_df = pd.DataFrame(temp)
    concat_df = pd.concat([df,temp_df])
    return concat_df


# In[34]:


def augment_data(df,cond,n):
    ds = np.asarray(df.loc[cond,'input'].tolist()).astype(np.float32)
    labels = np.asarray(df.loc[cond,df.columns[1:]]).astype(np.float32)
    data_aug = data_augmentation(ds,labels,n)
    return assign_ds_to_df(df,data_aug)


# In[35]:


def augment_by_factor(df,cond,n,factor):
    augmented_df = df.copy()
    for i in range(0,factor):
        augmented_df = augment_data(augmented_df,cond,n)
    return augmented_df


# ### Wasp Augmentation

# In[36]:


bee_wasp_df = augment_by_factor(augment_bee_df,(augment_bee_df['output/wasps_output'] == 1.0),600,3).reset_index(drop=True)


# In[37]:


plot_class_counts(bee_wasp_df,(bee_wasp_df['output/wasps_output'] == 1.0),'Number of images of bees and wasps',['bees','wasps'])


# Varroa Augmentation

# In[80]:


#Remove wasps
varroa_df = augment_bee_df.drop(augment_bee_df.loc[augment_bee_df['output/wasps_output']== 1.0].index).reset_index(drop=True)
#varroa_df_IVS = bee_df_IVS.drop(bee_df_IVS.loc[bee_df_IVS['output/wasps_output']== 1.0].index).reset_index(drop=True)
#Augment Data on train
varroa_df = augment_by_factor(varroa_df,(varroa_df['output/varroa_output'] == 1.0),800,1).reset_index(drop=True)


# In[81]:


plot_class_counts(varroa_df,(varroa_df['output/varroa_output'] == 1.0),'Number of images of bees sick with varroa and healthy',['varroa_no','varroa_yes'])


# Bee job train Augmentation

# In[40]:


#Remove wasps
multi_df = augment_bee_df.drop(augment_bee_df.loc[augment_bee_df['output/wasps_output']== 1.0].index).reset_index(drop=True)
#multi_df_IVS = bee_df_IVS.drop(bee_df_IVS.loc[bee_df_IVS['output/wasps_output']== 1.0].index).reset_index(drop=True)
#Augment Data on train
multi_df = augment_by_factor(multi_df,(multi_df['output/cooling_output'] == 1.0),700,2).reset_index(drop=True)
multi_df = augment_by_factor(multi_df,(multi_df['output/pollen_output'] == 1.0),600,2).reset_index(drop=True)
multi_df = augment_by_factor(multi_df,(multi_df['output/pollen_output'] == 1.0 )&\
                                   (multi_df['output/cooling_output'] == 1.0),90,2).reset_index(drop=True)


# In[42]:


plot_class_counts(multi_df,(multi_df['output/cooling_output'] == 1.0),'Number of images of bees performing cooling',['cooling_no','cooling_yes'])


# In[43]:


plot_class_counts(multi_df,(multi_df['output/pollen_output'] == 1.0),'Number of images of bees with and without polen',['polen_no','polen_yes'])


# ## Classification

# #### Split into train and independet validation set

# In[44]:


def split_dataset(df):
    X_train, X_IVS, y_train, y_test_IVS = train_test_split(df['input'].values,df[df.columns[1:]].values,test_size=0.33, random_state=42)
    return X_train, X_IVS, y_train, y_test_IVS


# In[45]:


def create_df(X,y):
    df = pd.DataFrame({'input':X})
    df['output/cooling_output'] = y[:,0]
    df['output/pollen_output'] = y[:,1]
    df['output/varroa_output'] = y[:,2]
    df['output/wasps_output'] =  y[:,3]
    return df


# In[46]:


def reshape_dataset(train,test,cond_train,cond_test,target,isSeries=True):
    X_train = np.asarray(list(train.loc[cond_train,'input'])).astype(np.float32)
    X_IVS = np.asarray(list(test.loc[cond_test,'input'])).astype(np.float32)

    if isSeries:
        y_train = np.asarray(list(train.loc[cond_train,target])).astype(np.float32)
        y_IVS = np.asarray(list(test.loc[cond_test,target])).astype(np.float32)
    else:
        y_train = np.asarray(list(train.loc[cond_train,target].values)).astype(np.float32)
        y_IVS = np.asarray(list(test.loc[cond_test,target].values)).astype(np.float32)

    #return train_ds,test_ds

    return X_train,X_IVS,y_train,y_IVS


# ### Dataset For Binary Classification Bee VS Wasp

# In[ ]:


X_train_bee,X_IVS_bee,y_train_bee,y_IVS_bee = split_dataset(bee_wasp_df)
bee_wasp_df_train = create_df(X_train_bee,y_train_bee)
bee_df_IVS = create_df(X_IVS_bee,y_IVS_bee)


# In[ ]:


cond_train = (bee_wasp_df_train['output/wasps_output'] == 0.0) | (bee_wasp_df_train['output/wasps_output'] == 1.0)
cond_test = (bee_df_IVS['output/wasps_output'] == 0.0) | (bee_df_IVS['output/wasps_output'] == 1.0)
#Creation Dataset
#X_train_bee_wasp,y_train_bee_wasp,X_test_bee_wasp,y_test_bee_wasp = create_ds(bee_df_train,bee_df_test,cond_train,cond_test,'output/wasps_output')

#X_train_bee,y_train_bee = (np.asarray(list(bee_df_train.loc[:,'input'])).astype(np.float32),np.asarray(list(bee_df_train.loc[:,'output/wasps_output'])).astype(np.float32))

X_train_bee,X_IVS_bee,y_train_bee,y_IVS_bee = reshape_dataset(bee_wasp_df_train,bee_df_IVS,cond_train,cond_test,'output/wasps_output')


# ### Dataset For Binary Classification Varroa VS No Varroa

# In[82]:


X_train_varroa,X_IVS_varroa,y_train_varroa,y_IVS_varroa = split_dataset(varroa_df)
varroa_df_train = create_df(X_train_varroa,y_train_varroa)
varroa_df_IVS = create_df(X_IVS_varroa,y_IVS_varroa)


# In[83]:


cond_train = (varroa_df_train['output/varroa_output'] == 0.0) | (varroa_df_train['output/varroa_output'] == 1.0)
cond_test = (varroa_df_IVS['output/wasps_output'] == 0.0) | (varroa_df_IVS['output/wasps_output'] == 1.0)
#Creation Dataset
#X_train_varroa,y_train_varroa,X_test_varroa,y_test_varroa= create_ds(bee_df_train,bee_df_test,cond,'output/varroa_output')
X_train_varroa,X_IVS_varroa,y_train_varroa,y_IVS_varroa = reshape_dataset(varroa_df_train,varroa_df_IVS,cond_train,cond_test,'output/varroa_output')


# ### Dataset For Multi Classe Classification  Bee Job

# In[47]:


X_train_multi,X_IVS_multi,y_train_multi,y_IVS_multi = split_dataset(multi_df)
multi_df_train = create_df(X_train_multi,y_train_multi)
multi_df_IVS = create_df(X_IVS_multi,y_IVS_multi)


# In[50]:


cond_train = (multi_df_train['output/cooling_output'] == 0.0) | (multi_df_train['output/cooling_output'] == 1.0)| \
(multi_df_train['output/pollen_output'] == 0.0) | (multi_df_train['output/pollen_output'] == 1.0)

cond_test = (multi_df_IVS['output/cooling_output'] == 0.0) | (multi_df_IVS['output/cooling_output'] == 1.0)| \
(multi_df_IVS['output/pollen_output'] == 0.0) | (multi_df_IVS['output/pollen_output'] == 1.0)

#Creation Dataset
#train_ds_bee_job,val_ds_bee_job = create_ds(bee_df_train,bee_df_test,cond,bee_df_train.columns[1:3],False)
X_train_multi,X_IVS_multi,y_train_multi,y_IVS_multi= reshape_dataset(multi_df_train,multi_df_IVS,cond_train,cond_test,multi_df_train.columns[1:3],False)


# In[51]:


def create_multi_target_var(df):
    for i, val in enumerate(df.values[:,1:3]):
        if val[0] == 0.0 and val[1] == 0.0:
            df.loc[i,'bee_job'] = 0.0
        elif val[0] == 1.0 and val[1] == 0.0:
            df.loc[i,'bee_job'] = 1.0
        elif val[0] == 0.0 and val[1] == 1.0:
            df.loc[i,'bee_job'] = 2.0
        elif val[0] == 1.0 and val[1] == 1.0:
            df.loc[i,'bee_job'] = 3.0
            
    return df['bee_job']


# In[52]:


multi_df_train['bee_job'] = create_multi_target_var(multi_df_train)
multi_df_IVS['bee_job'] = create_multi_target_var(multi_df_IVS)


# In[53]:


#Create target for XGBoost
y_train_bee_job_XGB = np.asarray(list(multi_df_train.loc[(multi_df_train['output/cooling_output'] == 0.0) | \
                                                       (multi_df_train['output/cooling_output'] == 1.0) | \
                                                       (multi_df_train['output/pollen_output'] == 0.0) | \
                                                       (multi_df_train['output/pollen_output'] == 1.0),'bee_job'])).astype(np.float32)

y_IVS_bee_job_XGB = np.asarray(list(multi_df_IVS.loc[(multi_df_IVS['output/cooling_output'] == 0.0) | \
                                                    (multi_df_IVS['output/cooling_output'] == 1.0) | \
                                                    (multi_df_IVS['output/pollen_output'] == 0.0) | \
                                                    (multi_df_IVS['output/pollen_output'] == 1.0),'bee_job'])).astype(np.float32)


# TRAIN MODELS
# CNN - Exemplo da aula
# 

# Se não tiver Dense como é que faz o backpropogation?
# 
# 

# HyperParams:
# 
# - Batch Size
# - Kernel number/size
# - Numero de camadas / Neuronios por Camada
# - 

# In[54]:


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


# Cross-validation

# In[55]:


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


# In[56]:


tf.config.experimental_run_functions_eagerly(True)


# In[57]:


def cv(X,y,params,cv_splits,batch_sz=64,epochs=10,loss=None,dims=None,isNN=True):
  #***CNN****
  if isNN:
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
      early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=loss,
                metrics=['accuracy'])
      #Save initial weights to reset after each fold
      model.save_weights('initial_weights.h5')
      print(model.summary())
      i=0
      for train_idx,test_idx in kf.split(X):
        print("Fold number:",i)
        #Reset Weights
        model.load_weights('initial_weights.h5')
        #get train-test set
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        #batch_values = np.arange(50,150,25)
        #for batch in batch_values:
        #Fit model
        history = model.fit(
          x=X_train,
          y=y_train,
          validation_data=(X_test,y_test),
          epochs=epochs,
          batch_size = batch_sz,
          callbacks=early_stopping
        )
        i+=1
        history_models.append({'model_name':model.name,'model_def':model_def,'history':history})
    return history_models
  #***XGBOOST******
  else:
    xgb = XGBClassifier(eval_metric='error')
    gcv = GridSearchCV(xgb,params,cv=cv_splits)
    gcv.fit(X,y)
    return gcv.cv_results_


# In[58]:


def get_scores(history,n_splits):
    hist_df = pd.DataFrame(history)
    scores = list()
    for i in range(n_splits,len(history)+n_splits,n_splits):
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


# In[ ]:


# Run Time export class
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



# In[59]:


def plot_train_val(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()


# In[60]:


def select_best_model(scores):
    best_model = int(scores.loc[scores['val_loss_mean'] == scores['val_loss_mean'].min()]['model_name'].values[0].split("_")[1])
    return best_model


# In[61]:


def train_xgb(X,y,n_splits,params,obj_func):
    results = cv(X,y,params,n_splits,isNN=False)
    best_params = results['params'][np.where(results['mean_test_score'].max())[0][0]]
    max_depth = best_params['max_depth']
    eta = best_params['eta']
    gamma = best_params['gamma']
    xgb_model = XGBClassifier(objective=obj_func,max_depth=max_depth,eta=eta,gamma=gamma,eval_metric='error')
    xgb_model.fit(X,y)
    return xgb_model,results


# In[62]:


def plot_confusion_matrix(y_test,y_pred,model_name):
    plt.figure(figsize=(5,5))
    #cm = confusion_matrix(y_test,y_pred)
    cm = tf.math.confusion_matrix(y_test,y_pred)
    hmap = sns.heatmap(cm,cbar=False,annot=True,cmap="Blues",fmt='.0f')
    hmap.set_title('Confusion Matrix {}'.format(model_name))
    hmap.set_xlabel('predicted label')
    hmap.set_ylabel('true label')
    plt.show()


# ### Bee VS Wasp Model

# In[57]:


params_bee = {'filters__Conv2D':[[16,32,64,128],[8,16,32,64]],
          'kernel_size__Conv2D': [2,3],
          'momentum__BatchNormalization':[0.2],
          'epsilon__BatchNormalization':[1e-4],
          'pool_size__MaxPooling2D':[2],
          'rate__Dropout':[0.2,0.4],
          'units__Dense': [1],
          'activation__Dense': ['sigmoid'],
          'name__Flatten':['flatten_layer']}
dims = (150,75,3)


# In[144]:


n_splits = 4
loss = tf.keras.losses.BinaryCrossentropy()
histories = cv(X_train_bee,y_train_bee,params_bee,n_splits,64,5,loss,dims)


# In[145]:


scores_df = pd.DataFrame(get_scores(histories,n_splits))
scores_df


# In[146]:


models_def = grid_search(params_bee)
idx_model = select_best_model(scores_df)
model_bee = create_model(models_def[idx_model],dims)
model_bee.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])


# In[147]:


model_bee.summary()


# In[148]:


X_train, X_test, y_train, y_test = train_test_split(X_train_bee,y_train_bee,test_size=0.33, random_state=0)


# In[149]:


epochs = 20
#epochs = 50
# runtime performances
time_callback = TimeHistory()

history = model_bee.fit(
  x=X_train,
  y=y_train,
  validation_data=(X_test,y_test),
  epochs=epochs,
  batch_size = 90,
  callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5), time_callback]
)


# In[150]:


#plot_train_val(history)


# In[ ]:


# runtime performances
print("wasp time callbacks")
print(time_callback.times)


# In[151]:


# Extraimos a camada intermédia
layer_name='flatten_layer'
intermediate_layer_model = Model(inputs=model_bee.input,
                                 outputs=model_bee.get_layer(layer_name).output)


# In[152]:


intermediate_output = intermediate_layer_model.predict(X_train_bee) 
intermediate_output = pd.DataFrame(data=intermediate_output)


# In[153]:


intermediate_output


# In[154]:

start = time.time()   
params_xgb = {'eta':[0.2,0.3],'gamma':[0,10],'max_depth':np.arange(3,4)}
xgb,results = train_xgb(intermediate_output,y_train_bee,4,params_xgb,'binary:logistic')
elapsed = time.time() - start
print("wasp xgboost runtime", elapsed)


# In[155]:


pd.DataFrame(results)


# #### Validating Model Bee VS Wasp

# #### CNN

# In[169]:


y_preds_bee = model_bee.predict(X_IVS_bee)
y_preds_bee = (y_preds_bee > 0.5).astype(int)
print("Independent Test Accuracy: ",model_bee.evaluate(X_IVS_bee,y_IVS_bee)[1])
print("Independent Test Precision: ",precision_score(y_IVS_bee,y_preds_bee))
print("Independent Test Recall: ",recall_score(y_IVS_bee,y_preds_bee))
#plot_confusion_matrix(y_IVS_bee,y_preds_bee,'CNN')


# #### CNN+XGBOOST

# In[158]:


intermediate_test_output = intermediate_layer_model.predict(X_IVS_bee)
intermediate_test_output = pd.DataFrame(data=intermediate_test_output)


# In[170]:


y_preds_bee = xgb.predict(intermediate_test_output)
print("Independent Test Accuracy: ",xgb.score(intermediate_test_output,y_IVS_bee))
print("Independent Test Precision: ",precision_score(y_IVS_bee,y_preds_bee))
print("Independent Test Recall: ",recall_score(y_IVS_bee,y_preds_bee))
#plot_confusion_matrix(y_IVS_bee,y_preds_bee,'CNN+XGBoost')


# ## Varroa Model

# In[84]:


n_splits = 4
loss = tf.keras.losses.BinaryCrossentropy()
params_varroa = {'filters__Conv2D':[[16,32,64,128],[8,16,32,64]],
          'kernel_size__Conv2D': [2,4],
          'momentum__BatchNormalization':[0.2],
          'epsilon__BatchNormalization':[1e-4],
          'pool_size__MaxPooling2D':[2],
          'rate__Dropout':[0.2,0.4],
          'units__Dense': [1],
          'activation__Dense': ['sigmoid'],
          'name__Flatten':['flatten_layer']}
histories_varroa = cv(X_train_varroa,y_train_varroa,params_varroa,n_splits,90,10,loss,dims)


# In[85]:


scores_df_varroa = pd.DataFrame(get_scores(histories_varroa,n_splits))
scores_df_varroa


# In[86]:


models_def = grid_search(params_varroa)
idx_model = select_best_model(scores_df_varroa)
model_varroa = create_model(models_def[idx_model],dims)
model_varroa.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X_train_varroa,y_train_varroa,test_size=0.33, random_state=0)


# In[88]:


epochs = 15
#epochs = 50
# runtime performances
time_callback = TimeHistory()

history_varroa = model_varroa.fit(
  x=X_train,
  y=y_train,
  validation_data=(X_test,y_test),
  epochs=epochs,
  batch_size = 90,
  callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5), time_callback]  
)
# runtime performances
print("wasp time callbacks")
print(time_callback.times)

# In[89]:


#plot_train_val(history_varroa)


# In[90]:


# Extraimos a camada intermédia
layer_name='flatten_layer'
intermediate_layer_model = Model(inputs=model_varroa.input,
                                 outputs=model_varroa.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(X_train_varroa) 
intermediate_output = pd.DataFrame(data=intermediate_output)
intermediate_output


# In[91]:

start = time.time() 
params_xgb = {'eta':[0.2,0.3],'gamma':[0,10],'max_depth':np.arange(3,4)}
xgb_varroa,results = train_xgb(intermediate_output,y_train_varroa,4,params_xgb,'binary:logistic')
pd.DataFrame(results)
elapsed = time.time() - start
print("varroa xgboost runtime", elapsed)

# #### Validating Model Varroa

# #### CNN

# In[92]:


y_preds_varroa = model_varroa.predict(X_IVS_varroa)
y_preds_varroa = (y_preds_varroa > 0.5).astype(int)
print("Independent Test Accuracy: ",model_varroa.evaluate(X_IVS_varroa,y_IVS_varroa)[1])
print("Independent Test Precision: ",precision_score(y_IVS_varroa,y_preds_varroa))
print("Independent Test Recall: ",recall_score(y_IVS_varroa,y_preds_varroa))
#plot_confusion_matrix(y_IVS_varroa,y_preds_varroa,'CNN')


# #### CNN+XGBOOST

# In[93]:


intermediate_test_output = intermediate_layer_model.predict(X_IVS_varroa)
intermediate_test_output = pd.DataFrame(data=intermediate_test_output)
y_preds_varroa = xgb_varroa.predict(intermediate_test_output)
print("Independent Test Accuracy: ",xgb_varroa.score(intermediate_test_output,y_IVS_varroa))
print("Independent Test Precision: ",precision_score(y_IVS_varroa,y_preds_varroa))
print("Independent Test Recall: ",recall_score(y_IVS_varroa,y_preds_varroa))
#plot_confusion_matrix(y_IVS_varroa,y_preds_varroa,'CNN+XGBoost')


# ## Multi-job Model

# In[64]:


n_splits = 4
dims= (150,75,3)
loss = tf.keras.losses.BinaryCrossentropy()
params_multi = {'filters__Conv2D':[[8,16,32,64]], # [16,32,64,128],
          'kernel_size__Conv2D': [2,4],
          'momentum__BatchNormalization':[0.2],
          'epsilon__BatchNormalization':[1e-3], # [1e-4]
          'pool_size__MaxPooling2D':[2],
          'rate__Dropout':[0.2,0.4],
          'units__Dense': [2],
          'activation__Dense': ['sigmoid'],
          'name__Flatten':['flatten_layer']}

histories_multi = cv(X_train_multi,y_train_multi,params_multi,n_splits,32,10,loss,dims)


# In[65]:


scores_df_multi = pd.DataFrame(get_scores(histories_multi,n_splits))
scores_df_multi


# In[66]:


models_def = grid_search(params_multi)
idx_model = select_best_model(scores_df_multi)
model_multi = create_model(models_def[idx_model],dims)
model_multi.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])
model_multi.summary()


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X_train_multi,y_train_multi,test_size=0.33, random_state=0)


# In[68]:


epochs = 20
#epochs = 50
time_callback = TimeHistory()

history_multi = model_multi.fit(
  x=X_train,
  y=y_train,
  validation_data=(X_test,y_test),
  epochs=epochs,
  batch_size = 32,
  callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5), time_callback]
)

# runtime performances
print("wasp time callbacks")
print(time_callback.times)
# In[69]:


#plot_train_val(history_multi)


# In[70]:


# Extraimos a camada intermédia
layer_name='flatten_layer'
intermediate_layer_model = Model(inputs=model_multi.input,
                                 outputs=model_multi.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(X_train_multi) 
intermediate_output = pd.DataFrame(data=intermediate_output)
intermediate_output


# In[72]:

start = time.time()
params_xgb = {'eta':[0.2,0.3],'gamma':[0,10],'max_depth':np.arange(3,4)}
xgb_multi,results = train_xgb(intermediate_output,y_train_bee_job_XGB,4,params_xgb,'multi:softmax')
pd.DataFrame(results)
elapsed = time.time() - start
print("multi xgboost runtime", elapsed)

# #### Validating Model Bee job

# #### CNN

# In[75]:


y_preds_multi = model_multi.predict(X_IVS_multi)
y_preds_multi = (y_preds_multi > 0.5).astype(int)
print("Independent Test Accuracy: ",model_multi.evaluate(X_IVS_multi,y_IVS_multi)[1])
#plot_confusion_matrix(y_IVS_multi,y_preds_multi,'CNN')


# #### CNN+XGBOOST

# In[79]:


intermediate_test_output = intermediate_layer_model.predict(X_IVS_multi)
intermediate_test_output = pd.DataFrame(data=intermediate_test_output)
y_preds_multi = xgb_multi.predict(intermediate_test_output)
print("Independent Test Accuracy: ",xgb_multi.score(intermediate_test_output,y_IVS_bee_job_XGB))
plot_confusion_matrix(y_IVS_bee_job_XGB,y_preds_multi,'CNN+XGBoost')


# ### Outro dataset de abelhas

# In[ ]:










