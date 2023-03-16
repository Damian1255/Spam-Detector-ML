import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the model
model = load_model('Models/RNNmodel.h5')

# Get input text for prediction
text = input('Enter your message: ')

# Preprocess the text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])
X = pad_sequences(sequences, maxlen=50)

# Predict
y_pred = model.predict(X)
y_pred = np.argmax(y_pred, axis=1)

# Print result
if y_pred == 1:
    print('Spam')
else:
    print('Not Spam')
