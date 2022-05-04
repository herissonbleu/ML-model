from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from keras.losses import MeanSquaredError
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.layers import IntegerLookup
from keras.layers import Normalization
from keras.layers import StringLookup
import json
from tensorflow.python.framework.ops import disable_eager_execution

"""
This module is using data from multiple years with missing values filled accordingly

"""
#disable_eager_execution()

df = pd.read_csv(r'path', delimiter=',',keep_default_na=False)

#Next part is data specific
#df.drop(df.index[df['colname'] == None], inplace=True)
#df.drop(df.index[df['colname'] == None],inplace=True)
#df.drop('colname',axis=1,inplace=True)
#df.drop('colname',axis=1,inplace=True)
#df.drop('colname',axis=1,inplace=True)
#df.drop('colname',axis=1,inplace=True)
#df.drop('colname',axis=1,inplace=True)
df['colname'].astype(int)

di = {1:1,2:1,3:0,4:0,5:0}
df.replace({"colname": di},inplace=True)

val_dataframe = df.sample(frac=0.2,random_state=1337)
train_dataframe = df.drop(val_dataframe.index)
print("Using %d samples for training and %d for validation" % (len(train_dataframe), len(val_dataframe)))

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("colname").astype(int)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x,y in train_ds.take(1):
    print("Input: ",x)
    print("Target: ",y)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

def encode_numerical_feature(feature,name,dataset):
    normalizer = Normalization()
    feature_ds = dataset.map(lambda x,y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x,-1))
    normalizer.adapt(feature_ds)
    encoded_feature = normalizer(feature)
    return encoded_feature

def encode_categorical_feature(feature,name,dataset,is_string):
    lookup_class = layers.experimental.preprocessing.StringLookup if is_string else layers.experimental.preprocessing.IntegerLookup
    lookup = lookup_class(output_mode="binary")
    feature_ds = dataset.map(lambda x,y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x,-1))
    lookup.adapt(feature_ds)
    encoded_feature = lookup(feature)
    return encoded_feature

#Categorical features encoded as integers
cat1 = keras.Input(shape=(1,), name='cat1',dtype='int64')
cat2 = keras.Input(shape=(1,), name='cat2',dtype='string')

#numerical features
num1 = keras.Input(shape=(1,), name='num1')
num2 = keras.Input(shape=(1,), name='num2')
num3 = keras.Input(shape=(1,), name='num3')
num4 = keras.Input(shape=(1,), name='num4')
num5 = keras.Input(shape=(1,), name='num5')
#num6 = keras.Input(shape=(1,), name='num6')


all_inputs = [cat1,num1,cat2,
            num2,num3,num4,
            num5] #num6

#integer categorical features
cat1_encoded = encode_categorical_feature(cat1,'cat1',train_ds,False)
#String categorical features
cat2_encoded = encode_categorical_feature(cat2,'cat2',train_ds, True)

#Numerical features
num1_encoded = encode_numerical_feature(num1,'num1',train_ds)
num2_encoded = encode_numerical_feature(num2,'num2',train_ds)
num3_encoded = encode_numerical_feature(num3,'num3',train_ds)
num4_encoded = encode_numerical_feature(num4,'num4',train_ds)
num5_encoded = encode_numerical_feature(num5,'num5',train_ds)
#num6_encoded = encode_numerical_feature(num6,'num6',train_ds)

all_features = layers.concatenate(
    [cat1_encoded,cat2_encoded
    ,num1_encoded,
    num2_encoded,num3_encoded,num4_encoded,
    num5_encoded]#num6_encoded
)

x = layers.Dense(32,activation='elu',bias_initializer='zeros',kernel_initializer='random_normal')(all_features) #kernel_regularizer='l1' #bias_initializer='zeros',kernel_initializer='random_normal'
x = layers.Dropout(0.2)(x)
output = layers.Dense(units=1,activation='sigmoid')(x)
model = keras.Model(all_inputs,output)
model.compile('adam','binary_crossentropy',metrics=['accuracy'])

#'rankdir='LR' for horizontal graph
#keras.utils.plot.model(model,show_shapes=True, rankdir='LR')
history = model.fit(train_ds,epochs=50,validation_data=val_ds)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# serialize model to JSON

model_json = model.to_json()
with open("filename.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("filename as before",save_format='tf')
print("Saved model to disk")
