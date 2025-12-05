import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]
from sklearn.utils import class_weight
import tensorflow as tf # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Model # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Concatenate # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping # pyright: ignore[reportMissingImports]
from tensorflow.keras.initializers import Constant # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingImports]

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

data = pd.read_csv('tokenized_data.csv', engine='python')
X = data[['clean_text', 'clean_salary']]
y = data['fraudulent']  # target

train_idx = np.load('train_indices.npy')
test_idx = np.load('test_indices.npy')

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]
X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

max_words = 10000
max_len = 200

X_train_seq = tokenizer.texts_to_sequences(X_train['clean_text'])
X_test_seq = tokenizer.texts_to_sequences(X_test['clean_text'])
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

X_train_salary = np.array(X_train['clean_salary']).reshape(-1, 1)
X_test_salary = np.array(X_test['clean_salary']).reshape(-1, 1)

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# -----------------------------
# 1. Load GloVe embeddings
# -----------------------------
embedding_index = {}
glove_path = "C:\\Users\\Shirin Sabuwala\\Desktop\\Shirin\\MIT\\Sem 7\\Mini Project\\glove.6B.100d.txt\\glove.6B.100d.txt"

with open(glove_path, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

print(f"Loaded {len(embedding_index)} word vectors from GloVe.")

# -----------------------------
# 2. Build embedding matrix
# -----------------------------
embedding_dim = 100
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# -----------------------------
# 3. Define LSTM model
# -----------------------------
text_input = Input(shape=(max_len,), name='text_input')
embedding_layer = Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_len,
        trainable=True  # Set trainable to True for fine-tuning
    )(text_input)

#x = LSTM(32, return_sequences=False)(embedding_layer)
x = LSTM(64, return_sequences=False)(embedding_layer)
x = Dropout(0.5)(x)

salary_input = Input(shape=(1,), name='salary_input')

combined = Concatenate()([x, salary_input])
dense = Dense(64, activation='relu')(combined)
#model.add(Dense(32, activation="relu"))
dense = Dropout(0.3)(dense)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[text_input, salary_input], outputs=output)
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=1e-5),
    metrics=["accuracy"]
)

model.summary()

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# -----------------------------
# 4. Train with class weights
# -----------------------------
history = model.fit(
    [X_train_pad, X_train_salary],
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights,  # keep balancing the classes
    callbacks=[early_stop], # Add early stopping
    verbose=1
)

model.save('fake_job_classifier.keras')
print("Model trained and saved.")