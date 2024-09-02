import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'data')
    json_file_path = os.path.join(data_dir, 'problem_solutions.json')
    print(f"Resolved path: {json_file_path}")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def check_solution(problem_description, solution):
    return problem_description.lower() in solution.lower()

def train_model():
    df = load_data()

    texts = df['problem_description']
    solutions = df['gitlab_solution']
    packaging_types = df['packaging_type']
    labels = df['label']  # Use labels directly from the dataset

    # One-hot encode packaging types
    packaging_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    packaging_encoded = packaging_encoder.fit_transform(packaging_types.values.reshape(-1, 1))

    # Convert text features into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    text_features = vectorizer.fit_transform(texts)

    # Combine text features and packaging types
    X_text_packaging = hstack([text_features, packaging_encoded])

    # Check for the presence of both classes
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        raise ValueError(f"Not enough classes in the data: {unique_labels}")

    print("Unique Labels in Data:", unique_labels)
    print(f"Training Labels (y): {list(labels)}")
    print(f"Text Features Shape: {text_features.shape}")
    print(f"Packaging Encoded Shape: {packaging_encoded.shape}")

    # Train the model
    model = LogisticRegression()
    model.fit(X_text_packaging, labels)

    return model, vectorizer, packaging_encoder
