import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model('./Models/model_RNN_v2.h5')

# Load the tokenizer
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(open('./Dumps/tokenizer_RNN_v2.json').read())

# Define the maximum sequence length
max_sequence_length = 50

# Define the label encodings
labels = ['ham', 'spam']

# Ask the user for input
message = input("Enter a message: ")

# Tokenize the message
tokens = tokenizer.texts_to_sequences([message])

# Pad the token sequence
padded_tokens = pad_sequences(tokens, maxlen=max_sequence_length, padding='post')

# Make the prediction
prediction = model.predict([padded_tokens, np.zeros((1, len(labels)))])[0]

# Get the predicted label
predicted_label = labels[np.argmax(prediction)]

# Print the result
print(f"The message '{message}' is {predicted_label}")
