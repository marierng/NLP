# -*- coding: utf-8 -*-
"""Spam or ham Classification (NLP).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ePuqq3oLoo-7enZ-H2UYXpBN4txANUv9

# **About Dataset**
The SMS Spam Collection is a set of SMS messages that have been collected and labeled as either spam or not spam. This dataset contains 5574 English, real, and non-encoded messages. The SMS messages are thought-provoking and eye-catching. The dataset is useful for mobile phone spam research.

# **Importing Libraries**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

"""# **Observing the dataset**"""

df = pd.read_csv("train.csv")

df.head()

df['label']

df['text_length'] = df['sms'].apply(lambda x: len(x.split()))

"""# **Visualization**"""

plt.figure(figsize=(15,10),dpi=300)

"""## **Count Plot and Box Plot**








"""

plt.subplot(1,2,1)
# Count Plot
sns.countplot(x='label',data=df)
plt.title('Spam or Not Spam Distribution')
plt.xlabel('Spam or Not Spam')
plt.ylabel('Count')
# Box Plot
plt.subplot(1,2,2)
sns.boxplot(x='label', y ='text_length', data=df)
plt.title('Text Length Distribution')
plt.xlabel('Spam or Not Spam')
plt.ylabel('Text Length')
plt.tight_layout()
plt.show()

"""## **Text Length Distribution using Histplot**"""

sns.histplot(data=df,x = 'text_length', hue='label', kde= True,element='step')
plt.title('Histogram Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

"""## **Word Cloud for Spam and Ham**


"""

ham_text = " ".join(df[df['label'] == 0]['sms'])
spam_text = " ".join(df[df['label'] == 1]['sms'])

ham_wordcloud = WordCloud(width=800, height = 800 , background_color='pink').generate(ham_text)
spam_wordcloud = WordCloud(width = 800,height = 800, background_color = 'pink').generate(spam_text)

ham_image = ham_wordcloud.to_array()
spam_image = spam_wordcloud.to_array()

plt.subplot(1,2,1)
plt.imshow(ham_image,interpolation='bilinear')
plt.title('Ham Messages Word Cloud')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(spam_image,interpolation='bilinear')
plt.title('Spam Messages Word Cloud')
plt.axis('off')
plt.tight_layout()

spam_image

"""## **Distribution Plot**

"""

# Distribution Plot
sns.displot(data=df, x='text_length')
plt.title('Distribution Text Length')
plt.xlabel('Text Length')
plt.ylabel('Count (Rows)')
plt.show()

sns.kdeplot(data=df, x="label")
plt.title('Kde Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

sns.scatterplot(data=df, x="text_length", y="label")
plt.title("Scatterplot Distribution")
plt.xlabel("text_length")
plt.ylabel("label")
plt.show()

sns.ecdfplot(data=df, x="label")
plt.title('Kde Label Distribution')
plt.xlabel('Label')
plt.ylabel('Density')
plt.show()

"""# **Preprocessing**

* **Tokenization:** We will split each of the sentence into indivisual tokens. We are gonna make the split based to certain stop words such as spaces and punctuation marks.

* **Lowercasing:** All the words are converted to lower case.

* **Stopword Removal:** We eliminate common words that do not contribute anything.
"""

df.head()

def preprocessing_helper_function(sms):
  words = word_tokenize(sms) # tokenization
  words = [word.lower() for word in words if word.isalnum()] # Lowercasing
  words = [word for word in words if word not in stopwords.words('english')] # Removing Stopwords
  return " ".join(words)

#words = [word.lower() for word in words if word.isalnum()]
#for word in words:
 # if word.isalnum():
   #return word.lower()

#words = [word for word in words if word not in stopwords.words('english')]
#for word in words:
 # if word not in stopwords.words("english"):
   # return word()

df['sms'] = df['sms'].apply(preprocessing_helper_function)

"""# Tfidf ( Term Frequency Inverse Document Frequency) Vectorization"""

tfidf = TfidfVectorizer(max_features = 1000, ngram_range=(1,2))
X = tfidf.fit_transform(df['sms']).toarray() # Numpy
y = df['label']

"""# **Data Splitting**
Now that we have a preprocessed dataset for our feature that is `sms` and label `label`, now we can move on to training our model. But first we need a way to evaluate the performance of our model. So we will split the X and y further into training and test subsets. `test_size`
"""

x_train , x_test, y_train , y_test = train_test_split(X , y, test_size=0.20)

"""# **Model Training**

Model has three names
* Model
* Classifer
* Estimator
"""

model = MultinomialNB(alpha=.1,force_alpha=True)
model.fit(x_train,y_train)

"""###Optional but important to know

The part where we actually use our model to make predictions on real dta or outside data is called model inference.
"""

class SklearnNLTKClassifier(nltk.classify.ClassifierI):
  def __init__(self,classifier):
    self._classifier = classifier

  def classify(self, features):
    return self._classifier.predict([features])[0] # inference

  def classify_many(self,featuresset):
    return self._classifier.predict(featuresset) # For test dataset

  def prob_classify(self,features)  :
    raise NotImplementedError("Theres an error!") # For something that is impossible to classify

  def labels(self):
    return self.classifier.classes_   # Get the name of our labels

nltk_classifier = SklearnNLTKClassifier(model)

"""# **Evaluation**"""

y_pred = nltk_classifier.classify_many(x_test)

pred = model.predict(x_test)
print(classification_report(y_test,pred))

print(classification_report(y_test,y_pred))

con = confusion_matrix(y_test,y_pred)

sns.heatmap(con,annot=True,fmt='d',cmap='Reds', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

"""# **Testing on custom sentences**"""

def predict_spam(text):
  preprocessed_text = preprocessing_helper_function(text)
  transformed_text = tfidf.transform([preprocessed_text]).toarray()
  prediction = model.predict(transformed_text)
  return "Spam" if prediction[0] == 1 else "Not Spam"

sentence = 'Win free tickets !!'

print(predict_spam(sentence))

sentence2 = 'Lets meet on Sunday'
print(predict_spam(sentence2))

"""# **Simple Web App Inference of our Model**"""

!pip install streamlit

import joblib

joblib.dump(model,'model.pkl')
joblib.dump(tfidf,"tfidf.pkl")

!npm install localtunnel

!streamlit run app.py &>/content/logs.txt &

import urllib
print("Password/Enpoint IP for localtunnel is:",urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))

!npx localtunnel --port 8501

