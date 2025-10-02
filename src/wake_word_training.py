# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 19:49:23 2025

@author: sudha
"""
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Activation
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix,classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wake_word_config import all_configs
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)


n_mfcc = all_configs["feature_extraction"]["n_mfcc"]
# model Training

df = pd.read_pickle(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\Data\full_dataset_4.csv")
df=df[df["class"].isin([0])] # further training

x = df["feature"].values
y = df["class"].values

x= np.concatenate(x,axis=0).reshape(len(x),n_mfcc) 

y = to_categorical(list(y))
y = np.array([[1.,0.] for cat in y]) # further training


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = load_model(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\model\Hey_mike_trained_model_40_features_similar.h5")
model_temp = model
model = Sequential()
model.add(Dense(256,input_shape=x[0].shape))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(2,activation="softmax"))

model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


history = model.fit(x_train,y_train,epochs=100,batch_size=32,validation_split=0.2)
score = model.evaluate(x_test,y_test)
print("score: ",score)

y_pred = np.argmax(model.predict(x_test),axis=1) 

confusion_matri = confusion_matrix(np.argmax((y_test),axis=1), y_pred)
classification_report(np.argmax((y_test),axis=1),y_pred)
plt.figure()
sns.heatmap(data=confusion_matri)
plt.show()
model.save(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\model\Hey_mike_trained_model_40_features_similar_FT.h5")


