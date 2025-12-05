import pickle
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re, string
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]

#nltk.download('wordnet')
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

def predict_job(title, company, description, requirements, benefits, salary):
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = load_model('fake_job_classifier.keras')

    input_text = f"{title} {company} {description} {requirements} {benefits}"
    input_cleaned = clean_text(input_text)
    input_seq = tokenizer.texts_to_sequences([input_cleaned])
    input_pad = pad_sequences(input_seq, maxlen=200, padding='post', truncating='post')

    input_salary = clean_salary(salary)
    if pd.isna(input_salary):
        input_salary = 0.0

    score = model.predict([input_pad, np.array([[input_salary]])])
    prediction = (score > 0.48).astype("int32")
    return score, prediction[0][0]
    #return prediction

'''# Example usage:
score, result = predict_job(
    title="Virtual Assistant",
    company="Multi-Billion Dollar",
    description="Manage finances and meetings",
    requirements="Internet Connection",
    benefits="",
    salary="₹12"
)

print("Predict Score: ", score)
print("Prediction: ", "Real" if result == 0 else "Fake")'''