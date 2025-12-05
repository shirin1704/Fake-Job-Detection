import pandas as pd
import numpy as np
import pickle
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv('combined_job_postings.csv', engine='python')
data.fillna('', inplace=True)
data['text'] = data[['title', 'description', 'company_profile', 'requirements', 'benefits']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
scaler = MinMaxScaler()

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'\d+', '', text)                      # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]  # remove stopwords
    tokens = [lemmatizer.lemmatize(w) for w in tokens]   # lemmatize
    return ' '.join(tokens)

def clean_salary(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = s.replace(',', '')  # remove commas
    s = re.sub(r'[^\d\-\–]', '', s)  # remove non-digit except hyphen
    # handle ranges like "50000-60000"
    if '-' in s or '–' in s:
        parts = re.split(r'[\-–]', s)
        parts = [p for p in parts if p.strip()]
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
    try:
        return float(s)
    except:
        return np.nan

data['clean_text'] = data['text'].apply(clean_text)
data['clean_salary'] = data['salary_range'].apply(clean_salary)
data['clean_salary'].fillna(data['clean_salary'].median(), inplace=True)
data['clean_salary'] = scaler.fit_transform(data[['clean_salary']])

X = data[['clean_text', 'clean_salary']]
y = data['fraudulent']  # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save('train_indices.npy', X_train.index)
np.save('test_indices.npy', X_test.index)

max_words = 10000   # size of vocabulary
max_len = 200       # max sequence length

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train['clean_text'])

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

tokenized_data = pd.DataFrame()
tokenized_data['clean_text'] = data['clean_text']
tokenized_data['clean_salary'] = data['clean_salary']
tokenized_data['fraudulent'] = data['fraudulent']
tokenized_data.to_csv('tokenized_data.csv', index=False)

print("Tokenizer and data saved.")