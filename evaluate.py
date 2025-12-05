import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]

model = load_model('fake_job_classifier.keras')
data = pd.read_csv('tokenized_data.csv', engine='python')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

X = data[['clean_text', 'clean_salary']]
Y = data['fraudulent']  # target

train_idx = np.load('train_indices.npy')
test_idx = np.load('test_indices.npy')

X_train = X.loc[train_idx]
y_train = Y.loc[train_idx]
X_test = X.loc[test_idx]
y_test = Y.loc[test_idx]

max_words = 10000
max_len = 200

X_train_seq = tokenizer.texts_to_sequences(X_train['clean_text'])
X_test_seq = tokenizer.texts_to_sequences(X_test['clean_text'])
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

X_train_salary = np.array(X_train['clean_salary']).reshape(-1, 1)
X_test_salary = np.array(X_test['clean_salary']).reshape(-1, 1)

loss, accuracy = model.evaluate([X_test_pad, X_test_salary], y_test, verbose=0)
print(f'Test Accuracy: {accuracy:.4f}')

# Predict on test data
y_pred_prob = model.predict([X_test_pad, X_test_salary])
y_pred = (y_pred_prob > 0.5).astype("int32")

# Classification report
#print("Classification Report (Fine-tuned GloVe):\n")
#print(classification_report(Y_test, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title("Confusion Matrix (Fine-tuned GloVe)")
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Compute precision, recall for all thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Compute average precision (area under PR curve)
ap_score = average_precision_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR curve (AP = {ap_score:.3f})", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve for Fake Job Classification")
plt.legend()
plt.grid(True)
plt.show()

# To find best balance point (max F1)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = f1_scores.argmax()
print(f"🔹 Best Threshold = {thresholds[best_idx]:.4f}")
print(f"Precision = {precision[best_idx]:.4f}, Recall = {recall[best_idx]:.4f}, F1 = {f1_scores[best_idx]:.4f}")


'''Test Accuracy: 0.9676

Best Threshold = 0.8960
Precision = 0.8280, Recall = 0.7514, F1 = 0.7879'''