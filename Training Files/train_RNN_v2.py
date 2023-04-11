import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

# Load the dataset
dataset = pd.read_csv('./Dataset/SMSSpamCollection_1000.txt', sep='\t', names=['label', 'message'])

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['message'])

# Save the tokenizer
with open('./Dumps/tokenizer_RNN_v2.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())

# Convert the text to sequences
sequences = tokenizer.texts_to_sequences(dataset['message'])
X = pad_sequences(sequences, maxlen=50)
y = pd.get_dummies(dataset['label']).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Load the pre-trained word embedding
embeddings_index = {}
with open(os.path.join('./Dumps', 'glove.6B.100d.txt'), encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create an embedding matrix using the pre-trained word embedding
word_index = tokenizer.word_index
num_words = len(word_index) + 1
embedding_dim = 100
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Use early fusion technique
label_input = Input(shape=(y.shape[1],))
text_input = Input(shape=(50,))
embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=50,
                            trainable=False)
x = embedding_layer(text_input)
x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
concat = Concatenate()([x, label_input])
output = Dense(3, activation='softmax')(concat)
model = Model(inputs=[text_input, label_input], outputs=output)

# Use a different optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_train, y_train], y_train, epochs=120, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate([X_test, y_test], y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the model
model.save('./Models/model_RNN_v2.h5')
