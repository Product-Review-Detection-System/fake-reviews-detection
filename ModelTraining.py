#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, \
    Bidirectional, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("data_df.csv", sep="\t")

# data['REVIEW_TEXT'] = data['REVIEW_TEXT'].apply(lambda x: " ".join(eval(x)))
# data.drop('PRODUCT_CATEGORY',axis=1,inplace=True)


data['RATING'] = data['RATING'].apply(lambda x: int(x > 3))


data["LABEL"] = data["LABEL"].apply(lambda x: int(x == 1))

W2V_SIZE = 48
W2V_WINDOW = 7
W2V_EPOCH = 64
W2V_MIN_COUNT = 5
w2v_model = gensim.models.word2vec.Word2Vec(
    window=W2V_WINDOW,
    min_count=W2V_MIN_COUNT,
    workers=8)

documents = []
for _text in data.REVIEW_TEXT:
    documents.append((_text.split(" ")))

w2v_model.build_vocab(documents)

# In[15]:


# w2v_model.most_similar("bad")

# In[17]:


np.mean(data.REVIEW_TEXT.map(len))

data.REVIEW_TEXT[0]

# In[17]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.REVIEW_TEXT)
vocab_size = len(tokenizer.word_index) + 1
print('Vocab Size is ', vocab_size)

# In[18]:


SEQUENCE_LENGTH = 180

# In[19]:


embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)

# In[20]:


ones = data[data["LABEL"] == 1]
twos = data[data["LABEL"] == 0]

# In[23]:


train_data = ones[:7500]
train_data = train_data.append(twos[:7500])
val_data = ones[7500:8000]
val_data = val_data.append(twos[7500:8000])
print(train_data.shape)
print(val_data.shape)

# In[44]:


test_data = ones[8000:]
test_data = test_data.append(twos[8000:])

data_nlp = train_data["REVIEW_TEXT"]
data_meta_input = train_data.drop(["REVIEW_TEXT","LABEL"],axis=1)
data_op = train_data["LABEL"]

val_nlp = val_data["REVIEW_TEXT"]
val_meta_input = val_data.drop(["REVIEW_TEXT","LABEL"],axis=1)
val_op = val_data["LABEL"]

test_data_nlp = test_data["REVIEW_TEXT"]
test_data_meta_input = test_data.drop(["REVIEW_TEXT","LABEL"],axis=1)
test_data_op = test_data["LABEL"]
x_data = pad_sequences(tokenizer.texts_to_sequences(data_nlp) , maxlen = SEQUENCE_LENGTH)
y_data = data_op
y_data = y_data.values.reshape(-1,1)
vx_data = pad_sequences(tokenizer.texts_to_sequences(val_nlp) , maxlen = SEQUENCE_LENGTH)
vy_data = val_op
vy_data = vy_data.values.reshape(-1,1)
testx = pad_sequences(tokenizer.texts_to_sequences(test_data_nlp) , maxlen = SEQUENCE_LENGTH)
testy = test_data_op
testy = testy.values.reshape(-1,1)


nlp_input = Input(shape=(SEQUENCE_LENGTH,))

emb = Embedding(vocab_size, 100, input_length=SEQUENCE_LENGTH)(nlp_input)
nlp_out = LSTM(64)(emb)
classifier1 = Dense(128, activation='relu')(nlp_out)
dropout = Dropout(0.2)(classifier1)
classifier2 = Dense(32, activation='relu')(dropout)
output = Dense(1, activation='sigmoid')(classifier2)

model = Model(inputs=[nlp_input], outputs=[output])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


history = model.fit(x_data, y_data, batch_size=32, epochs=3,
                    validation_data=(vx_data, vy_data))

# In[39]:


history2 = model.fit(x_data, y_data, batch_size=32, epochs=5, initial_epoch=3,
                     validation_data=(vx_data, vy_data))


model.save("Word2Vec.hdf5")



pred = model.predict(testx)
# np.unique(pred,return_counts=True)

pred = (pred > 0.5)
pred


# pred = pred.reshape(5000).astype(int)
# np.unique(pred, return_counts=True)


# **CONCATENATED MODEL**


# nlp_input = Input(shape=(SEQUENCE_LENGTH,))
# meta_input = Input(shape=(3,))
#
# emb = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)(nlp_input)
# nlp_out = LSTM(128)(emb)
#
# concat = concatenate([nlp_out, meta_input])
#
# classifier = Dense(128, activation='relu')(concat)
#
# output = Dense(1, activation='sigmoid')(classifier)
#
# model2 = Model(inputs=[nlp_input, meta_input], outputs=[output])
#
# # In[51]:
#
#
# model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # In[52]:
#
#
# model2.summary()
#
# # In[40]:
#
#
# historyx = model2.fit([x_data, data_meta_input], y_data, batch_size=32, epochs=3,
#                       validation_data=([vx_data, val_meta_input], vy_data))
#
#
#
# historyx = model2.fit([x_data, data_meta_input], y_data, batch_size=32, epochs=10, initial_epoch=3,
#                       validation_data=([vx_data, val_meta_input], vy_data))
#
#
# pred = model2.predict([testx, test_data_meta_input])
# pred = (pred > 0.5)
# pred
#
#
#
# pred = pred.reshape(5000).astype(int)
# np.unique(pred, return_counts=True)


