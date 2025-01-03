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
import gensim
from keras.initializers import Constant


messages = pd.read_csv("imdb_dataset.csv")

# messages = messages.iloc[:300,:]
corpus_words = []
corpus_sentences = []
WL =  WordNetLemmatizer()
for index in range(len(messages)):
  if index%1000==0:
     print(f"running index is {index}")
  html_tag_remover = re.sub("<.*>"," ",messages["review"][index])
  filtered_words = re.sub("\W"," ",html_tag_remover)
  message = filtered_words.lower()
  words_lst = [WL.lemmatize(word) for word in nltk.word_tokenize(message) if word not in stopwords.words('english')]
  document = " ".join(words_lst)
  corpus_sentences.append(document)
  corpus_words.append(words_lst)



  embed_dim = 300
  w2v_model = Word2Vec(sentences=corpus_words, vector_size=embed_dim, window=10, min_count=1)
  w2v_model.train(corpus_words, epochs=10, total_examples=len(corpus_words))
  vocab = w2v_model.wv.key_to_index

  print(len(vocab))

  vocab = list(vocab.keys())
  word_vec_dict = {}
  for word in vocab:
      word_vec_dict[word] = w2v_model.wv.get_vector(word)

  print(f"number of key-value pairs in dict is {len(word_vec_dict)}")

  max_num_tokens = -1
  for sentence in corpus_words:
      if len(sentence) > max_num_tokens:
          max_num_tokens = len(sentence)

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(corpus_sentences)
  vocab_size = len(tokenizer.word_index) + 1
  tokenized_sequences = tokenizer.texts_to_sequences(corpus_sentences)
  padded_sentences = pad_sequences(tokenized_sequences, maxlen=max_num_tokens, padding='post')

  print(f"first padded sentence is {padded_sentences[0]}")

  embed_matrix = np.zeros(shape=(vocab_size, embed_dim))
  for word, i in tokenizer.word_index.items():
      if word in word_vec_dict.keys():
          embed_vector = word_vec_dict[word]
          if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
              embed_matrix[i] = embed_vector

  print(f"first element in embed_matrix is:{embed_matrix[10][10]}")
  print("vocab_size is:", vocab_size)
  print("embed matrix shape is :", embed_matrix.shape)

  le = LabelEncoder()
  order = ["negative", "positive"]
  le.fit(order)
  Y = le.transform(messages['sentiment'])

  x_train, x_test, y_train, y_test = train_test_split(padded_sentences, Y, test_size=0.20, random_state=42)

  model = Sequential()
  model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_num_tokens,
                      embeddings_initializer=Constant(embed_matrix)))
  model.add(Bidirectional(LSTM(units=128)))
  model.add(Dense(64, activation="relu"))
  model.add(Dense(1, activation="sigmoid"))

  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

  print(model.summary())
  model.fit(x_train, y_train, epochs=3, verbose=1)
  pred = model.predict(x_test)

  pred = (pred > 0.5).astype(int)

  print(type(pred))
  print(pred[0])

  print(accuracy_score(y_test, pred))
  # print(accuracy_score(y_test,prediction))
  # print(f"accuracy_score:{accuracy}")
  # print(f"classification_report:{classification_report}")
  print(classification_report(y_test, pred))

