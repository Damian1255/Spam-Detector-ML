import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the model
model = load_model('Models/spam_RNN.h5')

while True:
    # Get input text for prediction
    text = input('Enter your message: ')

    # Preprocess the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    X = pad_sequences(sequences, maxlen=50)

    # Prediction
    y_pred = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)

    # Print result
    if y_pred == 1:
        print('Likely Spam\n')
    else:
        print('Unlikely Spam\n')

