import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load model & vectorizer
model = pickle.load(open('Dumps/spam.pkl', 'rb'))
cv = pickle.load(open('Dumps/vectorizer.pkl', 'rb'))

while True:
    # Get input text for prediction
    text = input('Enter your message: ')
    vect = cv.transform([text]).toarray()

    # Prediction
    my_prediction = model.predict(vect)

    # Prediction probability
    prob = model.predict_proba(vect)

    # Print result
    if my_prediction == 1:
        print(f'Likely Spam, Confidence Level: {prob[0][1]*100:.2f}%\n')
    else:
        print(f'Unlikely Spam, Confidence Level: {prob[0][0]*100:.2f}%\n')