from ml_project.api.model.train import train_model
from ml_project.api.model.predict import predict_solution

def test_predict_solution():
    model, vectorizer, packaging_encoder = train_model()
    result = predict_solution(model, vectorizer, packaging_encoder, "A web application for managing user data", "microservices")
    assert result == True