import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from scipy.sparse import hstack
import json
import pandas as pd
import os
import json

def load_data():
    # Get the directory of the current file (train.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the 'data' folder
    data_dir = os.path.join(script_dir, '..', '..', 'data')
    
    # Construct the path to the JSON file
    json_file_path = os.path.join(data_dir, 'problem_solutions.json')
    # Print the path for debugging
    print(f"Resolved path: {json_file_path}")
    # Open and load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def train_model():
    df = load_data()

    texts = df['problem_description']
    solutions = df['gitlab_solution']
    packaging_types = df['packaging_type']

    packaging_encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
    packaging_encoded = packaging_encoder.fit_transform(packaging_types.values.reshape(-1, 1))

    vectorizer = TfidfVectorizer()
    text_features = vectorizer.fit_transform(texts)

    X_text_packaging = hstack([text_features, packaging_encoded])

    def check_solution(problem_description, solution):
        return problem_description.lower() in solution.lower()

    y = [1 if check_solution(text, sol) else 0 for text, sol in zip(texts, solutions)]

    model = SVC(kernel='linear')
    model.fit(X_text_packaging, y)

    return model, vectorizer, packaging_encoder
