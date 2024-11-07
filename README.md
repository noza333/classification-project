# classification-project
# data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Handle missing values if any
    data.dropna(inplace=True)
    
    # Extract features and labels
    X = data['email_text']
    y = data['label']
    
    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, vectorizer
