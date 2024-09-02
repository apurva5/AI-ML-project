from scipy.sparse import hstack
import json

def predict_solution(model, vectorizer, packaging_encoder, problem_description, packaging_type):
    print("Script started")
    text_features_test = vectorizer.transform([problem_description])
    packaging_encoded_test = packaging_encoder.transform([[packaging_type]])
    X_test = hstack([text_features_test, packaging_encoded_test])
    print(f"Test Feature Vector: {X_test.toarray()}")  # Debugging line
    
    prediction = model.predict(X_test)
    print(f"Prediction: {prediction[0]}")  # Debugging line
    return prediction[0] == 1

   