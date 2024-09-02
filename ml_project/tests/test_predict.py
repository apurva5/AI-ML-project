from ml_project.api.model.train import train_model
from ml_project.api.model.predict import predict_solution
from scipy.sparse import hstack

def test_predict_solution():
    model, vectorizer, packaging_encoder = train_model()
    
    # Debugging the input transformations step-by-step
    problem_description = "A web application for managing user data"
    packaging_type = "microservices"

    # Vectorize problem description
    text_features_test = vectorizer.transform([problem_description])
    print(f"Text Features Test Shape: {text_features_test.shape}")
    print(f"Text Features Test Vector: {text_features_test.toarray()}")

    # Encode packaging type
    packaging_encoded_test = packaging_encoder.transform([[packaging_type]])
    print(f"Packaging Encoded Test Shape: {packaging_encoded_test.shape}")
    print(f"Packaging Encoded Test Vector: {packaging_encoded_test}")

    # Combine the features
    X_test = hstack([text_features_test, packaging_encoded_test])
    print(f"Combined Test Feature Vector: {X_test.toarray()}")

    # Perform prediction
    result = predict_solution(model, vectorizer, packaging_encoder, problem_description, packaging_type)
    print(f"Prediction Result: {result}")

    # Check the assertion
    assert result == True
