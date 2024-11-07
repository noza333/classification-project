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
    # model_selection.py

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def get_models():
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC()
    }
    return models
# model_training.py

def train_models(models, X_train, y_train):
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} model trained.")
    return trained_models
    # model_evaluation.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_test, y_test):
    # Predictions on test set
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1

def cross_validate_model(model, X_train, y_train):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    return cv_scores.mean()
