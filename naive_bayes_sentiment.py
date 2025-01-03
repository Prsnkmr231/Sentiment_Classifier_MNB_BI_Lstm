import pandas as pd
import numpy as np
import re
import nltk
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder


corpus = []
WL = WordNetLemmatizer()
PS = PorterStemmer()

messages = pd.read_csv("imdb_dataset.csv")

# messages = messages.iloc[:200,:]
# print(messages.isna().sum())
# print(messages.shape)
# print(messages.head(1))

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

#Using Bag of words model
# cv=CountVectorizer(max_features=3000)
# X=cv.fit_transform(corpus).toarray()


#Using TFIDF Vectorizer
tfv = TfidfVectorizer(max_features=3000)
X=tfv.fit_transform(corpus).toarray()

#Using pandas dummies function to encode the categorical variable into multiple columns
# based on the number of categorical variables.(if categorical variables=2 ,columns created=2)
# Y=pd.get_dummies(messages["sentiment"])
# Y = Y.iloc[:,1]

#Using LabelEncoder fucntion to encode the categorical variable into a single column
le = LabelEncoder()
order =  ["negative","positive"]
le.fit(order)
Y=le.transform(messages['sentiment'])
# Y=Y.values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=4)

NB_model = MultinomialNB(alpha=1.0, fit_prior=True)
NB_model.fit(X_train,Y_train)
Y_pred = NB_model.predict(X_test)

print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))





