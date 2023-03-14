import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# Importing the dataset from SMSSpamCollection_1000.txt
dataset = pd.read_csv('Dataset/SMSSpamCollection_1000.txt', sep='\t', names=['label', 'message'])

# Data preprocessing and cleaning
dataset['label'] = dataset.label.map({'ham': 0, 'spam': 1})
dataset = dataset.iloc[1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(dataset['message'], dataset['label'], test_size = 0.15, random_state = 0)


# Fitting the CountVectorizer to the Training set
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Fitting the MultinomialNB to the Training set
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification report
report = classification_report(y_test, y_pred)
print(report)

# Save model
pickle.dump(classifier, open('Dumps/spam.pkl', 'wb'))

# Save vectorizer
pickle.dump(cv, open('Dumps/vectorizer.pkl', 'wb'))






