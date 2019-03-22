#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import Input, Embedding, Dense
from keras.layers import GlobalMaxPool1D, GlobalMaxPool2D,Dropout
from keras.layers import concatenate
from keras.layers import  LSTM, CuDNNGRU, CuDNNLSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
import argparse
config = argparse.Namespace()

df_train = pd.read_csv('./train/train.csv',parse_dates=['activation_date'])
#筛选特征
df_train=df_train.drop(['image'],axis=1)#一定要加axis=1按列删
df_train=df_train.drop(['image_top_1'],axis=1)
df_y_train=df_train['deal_probability']
df_x_train=df_train.drop(['deal_probability'],axis=1)

eps = 1e-10
tr_price = np.log(df_x_train['price']+eps)
tr_price[tr_price.isnull()] = -1.

tr_price = np.expand_dims(tr_price, axis=-1)

def tknzr_fit(col,df_trn):
    tknzr = Tokenizer(filters='',lower=False,split='뷁')
    tknzr.fit_on_texts(df_trn[col])
    
    return np.array(tknzr.texts_to_sequences(df_trn[col])),tknzr


df_x_train['param_1'].fillna(value='_NA_', inplace=True)
df_x_train['param_2'].fillna(value='_NA_', inplace=True)
df_x_train['param_3'].fillna(value='_NA_', inplace=True)
df_x_train['description'].fillna(value='_NA_', inplace=True)
tr_reg,tknzr_reg = tknzr_fit('region',df_x_train)
tr_pcn, tknzr_pcn = tknzr_fit('parent_category_name', df_x_train )
tr_cn, tknzr_cn = tknzr_fit('category_name', df_x_train)
tr_ut,tknzr_ut = tknzr_fit('user_type', df_x_train)
tr_city, tknzr_city = tknzr_fit('city', df_x_train)
tr_p1, tknzr_p1 = tknzr_fit('param_1', df_x_train)
tr_p2, tknzr_p2 = tknzr_fit('param_2', df_x_train)
tr_p3, tknzr_p3 = tknzr_fit('param_3', df_x_train)
tr_week = pd.to_datetime(df_x_train['activation_date']).dt.weekday.astype(np.int32).values
tr_week = np.expand_dims(tr_week, axis=-1)
len_desc = 100000

tknzr_desc = Tokenizer(num_words=len_desc, lower='True')
tknzr_desc.fit_on_texts(df_x_train['description'].values)
tr_desc_seq = tknzr_desc.texts_to_sequences(df_x_train['description'].values)
maxlen= 75
tr_desc_pad = pad_sequences(tr_desc_seq, maxlen=maxlen)

len_title = 100000

tknzr_desc = Tokenizer(num_words=len_desc, lower='True')
tknzr_desc.fit_on_texts(df_x_train['title'].values)
tr_title_seq = tknzr_desc.texts_to_sequences(df_x_train['title'].values)
maxlen= 75
tr_title_pad = pad_sequences(tr_desc_seq, maxlen=maxlen)

X = np.array([tr_reg,tr_pcn,tr_ut,tr_city,tr_p1,tr_p2,tr_p3,tr_week,tr_price,tr_desc_pad,tr_title_pad])

Y = df_y_train

#注意这儿得到的X.shape= (特征种类，训练集行数)

valid_idx = df_y_train.sample(frac=0.2, random_state=1991).index#随机选择20%的数据作为验证集，得到的是行索引

train_idx = df_y_train[np.invert(df_y_train.index.isin(valid_idx))].index#剩下的作为训练集

​

X_train = [x[train_idx] for x in X]#利用得到的列index进行划分数据集。这里的列与训练集的行数是一致的

X_valid = [x[valid_idx] for x in X]

​

Y_train = Y[train_idx]

Y_valid = Y[valid_idx]

​

X_train.append(tr_desc_pad[train_idx])#处理文本类数据

​

X_valid.append(tr_desc_pad[valid_idx])

​

batch_size=4096
epochs=20
inp_reg = Input(shape=(1, ), name='inp_region')
emb_reg=Embedding(len(tknzr_reg.word_index)+1,8,input_shape=(1,))(inp_reg)

inp_pcn = Input(shape=(1, ), name='inp_parent_category_name')
emb_pcn = Embedding(len(tknzr_pcn.word_index)+1, 4, name='emb_parent_category_name')(inp_pcn)

inp_cn = Input(shape=(1, ), name='inp_category_name')
emb_cn = Embedding(len(tknzr_reg.word_index)+1,8 , name="emb_category_name" )(inp_cn)

inp_ut = Input(shape=(1, ), name='inp_user_type')
emb_ut = Embedding(len(tknzr_ut.word_index)+1,2, name='emb_user_type' )(inp_ut)

inp_city = Input(shape=(1, ), name='inp_city')
emb_city=Embedding(len(tknzr_city.word_index)+1,8,input_shape=(1,))(inp_city)

inp_p1 = Input(shape=(1, ), name='inp_p1')
emb_p1 = Embedding(len(tknzr_p1.word_index)+1,8, name='emb_p1')(inp_p1)

inp_p2 = Input(shape=(1, ), name='inp_p2')
emb_p2 = Embedding(len(tknzr_p2.word_index)+1, 16, name='emb_p2')(inp_p2)

inp_p3 = Input(shape=(1, ), name='inp_p3')
emb_p3 = Embedding(len(tknzr_p3.word_index)+1,16, name='emb_p3')(inp_p3)


inp_week = Input(shape=(1, ), name='inp_week')
emb_week = Embedding(7, 4, name='emb_week' )(inp_week)
conc_cate = concatenate([emb_reg, emb_pcn, emb_ut, emb_city, emb_p1, emb_p2, emb_p3, emb_week], axis=-1, name='concat_categorcal_vars')
#conc_cate = concatenate([emb_reg, emb_pcn, emb_cn], axis=-1, name='concat_categorcal_vars')

conc_cate = GlobalMaxPool1D()(conc_cate)#池化层，对时域1D信号进行最大值池化

inp_price = Input(shape=(1, ), name='inp_price')
emb_price = Dense(16, activation='tanh', name='emb_price')(inp_price)

conc_cate = concatenate([conc_cate, emb_price], axis=-1)
x=Dense(200,activation='relu')(conc_cate)
x =Dropout(0.5)(x)
x=Dense(50,activation='relu')(x)
inp_desc = Input(shape=(maxlen, ), name='inp_desc')
emb_desc = Embedding(100000, 100, name='emb_desc')(inp_desc)


inp_title = Input(shape=(maxlen, ), name='inp_title')
emb_title = Embedding(100000, 100, name='emb_title')(inp_title)
title_layer = GRU(40, return_sequences=False)(emb_title)

conc_desc = concatenate([x, desc_layer,title_layer], axis=-1)
ｘ= BatchNormalization(epsilon=1e-6, weights=None)(conc_desc)
outp=Dense(1,activation='sigmoid')(x)
#print outp.shape
model = Model(inputs = [inp_reg,inp_pcn,inp_ut,inp_city,inp_p1,inp_p2,inp_p3,inp_week,inp_price],outputs = outp)、

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

model.compile(optimizer='adam', loss = root_mean_squared_error, metrics=[root_mean_squared_error])




checkpoint = ModelCheckpoint('./best.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
early = EarlyStopping(patience=3, mode='min')



#model.fit(X, Y, batch_size=batch_size, epochs=5, callbacks=[checkpoint], verbose=1)
model.fit(x=X_train, y=np.array(Y_train), validation_data=(X_valid, Y_valid), batch_size=4096, epochs=100, callbacks=[checkpoint,early], verbose=1)

#model.load_weights('best.hdf5')





