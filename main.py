import numpy as np
import pandas as pd
import tensorflow as tf
import keras

#from tensorflow.python.keras.models import Model
from keras.models import Model
#from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, Attention

from keras.layers import Input, Embedding, LSTM, Dense, Attention

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras_preprocessing import Tokenizer
#from tensorflow.python.keraspreprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load and preprocess your dataset (replace with your data loading and preprocessing)
#data = pd.read_csv('BobbleAI/valid1.csv') 
data = pd.read_json('BobbleAI/hin_valid.json',lines=True) # Load your dataset
english_texts = data['english word'].values
indic_texts = data['native word'].values


# Tokenizer = tf.keras.preprocessing()
# pad_sequences = tf.keras.preprocessing.sequence()


# Tokenize and pad sequences
tokenizer_english = Tokenizer(char_level=True)
tokenizer_indic = Tokenizer(char_level=True)

tokenizer_english.fit_on_texts(english_texts)
tokenizer_indic.fit_on_texts(indic_texts)

english_sequences = tokenizer_english.texts_to_sequences(english_texts)
indic_sequences = tokenizer_indic.texts_to_sequences(indic_texts)



max_seq_length = max(len(seq) for seq in english_sequences)

english_sequences = pad_sequences(english_sequences, maxlen=max_seq_length, padding='post')
indic_sequences = pad_sequences(indic_sequences, maxlen=max_seq_length, padding='post')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(english_sequences, indic_sequences, test_size=0.2, random_state=42)

# Build the sequence-to-sequence model with attention
input_layer = Input(shape=(max_seq_length,))
embedding_layer = Embedding(input_dim=len(tokenizer_english.word_index) + 1, output_dim=128)(input_layer)
encoder_lstm = LSTM(128, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(embedding_layer)
encoder_states = [state_h, state_c]

decoder_input_layer = Input(shape=(max_seq_length,))
decoder_embedding_layer = Embedding(input_dim=len(tokenizer_indic.word_index) + 1, output_dim=128)(decoder_input_layer)
decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding_layer, initial_state=encoder_states)

attention_layer = Attention()([encoder_outputs, decoder_outputs])
merged_output = keras.layers.concatenate([decoder_outputs, attention_layer])


output_layer = Dense(len(tokenizer_indic.word_index) + 1, activation='softmax')(merged_output)

model = Model([input_layer, decoder_input_layer], output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_train, y_train], y_train, validation_data=([X_val, y_val], y_val), epochs=10, batch_size=64)

# Inference
def translate_sentence(input_text):
    input_seq = tokenizer_english.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')
    predicted_sequence = []

    decoder_input = np.zeros((1, max_seq_length))

    for i in range(max_seq_length):
        output_probs = model.predict([input_seq, decoder_input])[0, i]
        predicted_char_index = np.argmax(output_probs)
        if predicted_char_index == 0:
            break  # Stop if the model predicts the end of the sequence character
        predicted_char = tokenizer_indic.index_word[predicted_char_index]
        predicted_sequence.append(predicted_char)
        decoder_input[0, i + 1] = predicted_char_index

    return ''.join(predicted_sequence)
    #return predicted_sequence
# Evaluate the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=-1)
    accuracy = accuracy_score(y.reshape(-1), y_pred.reshape(-1))
    f1 = f1_score(y.reshape(-1), y_pred.reshape(-1), average='macro')
    return accuracy, f1

accuracy, f1 = evaluate_model(model, [X_val, y_val], y_val)
print(f"Validation Accuracy: {accuracy}")
print(f"Validation F1-Score: {f1}")

# Test the model
#print(indic_sequences)
input_text = "chaand"
predicted_transliteration = translate_sentence(input_text)
print(f"Input: {input_text}")
print(f"Predicted Transliteration: {predicted_transliteration}")