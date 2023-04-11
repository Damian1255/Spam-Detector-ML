import pandas as pd
import pickle

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import string
from nltk.corpus import stopwords

# Importing the dataset
print('Importing the dataset...')
dataset = pd.read_csv('./Dataset/spam.csv', encoding='latin-1')

# Data cleaning & preprocessing
print('Data cleaning & preprocessing...')
dataset.dropna(how="any", inplace=True, axis=1)
dataset.columns = ['label', 'message']
dataset['label_num'] = dataset.label.map({'ham':0, 'spam':1})

# Function to remove punctuations
def text_process(mess):
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

dataset['clean_msg'] = dataset.message.apply(text_process)

# Splitting the dataset into the Training set and Test set
X = dataset.clean_msg
y = dataset.label_num
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Fitting the CountVectorizer to the Training set
cv = CountVectorizer(ngram_range=(1, 2), stop_words='english', max_df=0.5, min_df=2)
cv.fit(X_train)
X_train_dtm = cv.transform(X_train)
X_train_dtm = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)

# Normalize the data using tf-idf
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train_dtm)
tfidf_transformer.transform(X_train_dtm)

# Fitting the MultinomialNB to the Training set
print('Fitting Model...')
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# Predicting the Test set results
y_pred_class = nb.predict(X_test_dtm)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]

# Metrics
print('\n==================== Results ====================')
print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred_class)}')
print(f'Precision: {metrics.precision_score(y_test, y_pred_class)}')
print(f'AUC: {metrics.roc_auc_score(y_test, y_pred_prob)}')
print(f'Recall: {metrics.recall_score(y_test, y_pred_class)}')
print(f'F1 Score: {metrics.f1_score(y_test, y_pred_class)}')
print(f'Confusion Matrix:\n{metrics.confusion_matrix(y_test, y_pred_class)}')

# Save model & vectorizer
pickle.dump(nb, open('./Models/model_v2.pkl', 'wb'))
pickle.dump(cv, open('./Dumps/vectorizer_v2.pkl', 'wb'))

print('\nModel & vectorizer saved...')
