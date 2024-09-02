from ml_project.api.model.train import train_model
import pytest
def test_train_model():
    model, vectorizer, packaging_encoder = train_model()
    assert model is not None

    
    assert vectorizer is not None
    assert packaging_encoder is not None