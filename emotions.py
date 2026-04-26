import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv('C:/Users/GOURAV SHARMA/OneDrive/Documents/data-science-projects/NLP/train.txt', sep=';', header=None)
df.columns = ['text', 'emotions']
print(df.head())
print(df.isnull().sum())
le = LabelEncoder()
df['emotions'] = le.fit_transform(df['emotions'])
stop_words = set(stopwords.words('english'))
def clean_text(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = ''.join([i for i in txt if not i.isdigit()])
    txt = ''.join([i for i in txt if i.isascii()])
    words = word_tokenize(txt)
    return ' '.join([w for w in words if w not in stop_words])
df['text'] = df['text'].apply(clean_text)
x = df['text']
y = df['emotions']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
bow_vectorizer = CountVectorizer()
x_train_bow = bow_vectorizer.fit_transform(x_train)
x_test_bow = bow_vectorizer.transform(x_test)
nb_model = MultinomialNB()
nb_model.fit(x_train_bow, y_train)
prediction_nb = nb_model.predict(x_test_bow)
print("\n--- BoW Results ---")
print("Accuracy:", accuracy_score(y_test, prediction_nb))
print(classification_report(y_test, prediction_nb))
cm = confusion_matrix(y_test, prediction_nb)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("BoW Confusion Matrix")
plt.show()
tfidf_vectorizer = TfidfVectorizer()
x_train_tf = tfidf_vectorizer.fit_transform(x_train)
x_test_tf = tfidf_vectorizer.transform(x_test)
nb2_model = MultinomialNB()
param_grid = {'alpha': [0.1, 0.5, 1.0]}
grid = GridSearchCV(nb2_model, param_grid, cv=3)
grid.fit(x_train_tf, y_train)
best_model = grid.best_estimator_
pred2_nb = best_model.predict(x_test_tf)
print("\n--- TF-IDF Results ---")
print("Best Alpha:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, pred2_nb))
print(classification_report(y_test, pred2_nb))
cm2 = confusion_matrix(y_test, pred2_nb)
sns.heatmap(cm2, annot=True, fmt='d')
plt.title("TF-IDF Confusion Matrix")
plt.show()