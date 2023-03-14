import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load model
model = pickle.load(open('Dumps/spam.pkl', 'rb'))

# Load vectorizer
cv = pickle.load(open('Dumps/vectorizer.pkl', 'rb'))

while True:
    # Predict
    text = input('Enter your message: ')
    vect = cv.transform([text]).toarray()

    # Predict
    my_prediction = model.predict(vect)

    # Print result
    if my_prediction == 1:
        print('Spam')
    else:
        print('Not Spam')