import pandas as pd
import numpy as np
import re
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,Bidirectional,LSTM
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text  import Tokenizer,one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

messages = pd.read_csv("imdb_dataset.csv")

# messages = messages.iloc[:20,:]

corpus = []
WL =  WordNetLemmatizer()
for index in range(len(messages)):
  if index%1000==0:
     print(f"running index is {index}")
  html_tag_remover = re.sub("<.*>"," ",messages["review"][index])
  filtered_words = re.sub("\W"," ",html_tag_remover)
  message = filtered_words.lower()
  words_lst = [WL.lemmatize(word) for word in nltk.word_tokenize(message) if word not in stopwords.words('english')]
#   words_lst = [PS.stem(word) for word in nltk.word_tokenize(message) if word not in stopwords.words('english')]
  document = " ".join(words_lst)
  corpus.append(document)

le = LabelEncoder()
order =  ["negative","positive"]
le.fit(order)
Y=le.transform(messages['sentiment'])

X_train,X_test,Y_train,Y_test = train_test_split(corpus,Y,test_size=0.2,random_state=4)
# ##Using onehot vector representation for converting the sequences into vectors.

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
vocab_size = len(word_index)+1
sent_length = 100
emb_dim = 100
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences,padding="post",maxlen=sent_length)

test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences,padding="post",maxlen=sent_length)


model = Sequential()

model.add(Embedding(input_dim=vocab_size,output_dim=emb_dim,input_length=sent_length))
model.add(Bidirectional(LSTM(units=128)))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(train_padded,Y_train,epochs=10,verbose=1)
prediction = model.predict(test_padded)

print(prediction[:10])
for value in prediction:
    if value[0]>0.5:
        value[0]=1
    else:
        value[0]=0
print(accuracy_score(Y_test,prediction))
print(classification_report(Y_test,prediction))