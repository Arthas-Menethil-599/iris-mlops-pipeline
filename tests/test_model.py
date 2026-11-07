"""Unit tests for the Iris ML pipeline."""

import os
import tempfile
import pytest
from src.train import load_data, train_model, save_model
from src.predict import load_model, predict


class TestDataLoading:
    def test_load_data_shape(self):
        df, target_names = load_data()
        assert df.shape == (150, 5)
        assert len(target_names) == 3

    def test_load_data_columns(self):
        df, _ = load_data()
        assert "target" in df.columns
        assert len(df.columns) == 5

    def test_no_missing_values(self):
        df, _ = load_data()
        assert df.isnull().sum().sum() == 0


class TestTraining:
    def test_model_accuracy_above_threshold(self):
        df, _ = load_data()
        model, accuracy = train_model(df)
        assert accuracy > 0.90, f"Accuracy {accuracy} is below 0.90"

    def test_model_has_predict_method(self):
        df, _ = load_data()
        model, _ = train_model(df)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_model_reproducibility(self):
        df, _ = load_data()
        _, acc1 = train_model(df, random_state=42)
        _, acc2 = train_model(df, random_state=42)
        assert acc1 == acc2


class TestSaveLoad:
    def test_save_and_load(self):
        df, _ = load_data()
        model, _ = train_model(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            save_model(model, path)
            assert os.path.exists(path)
            loaded = load_model(path)
            assert hasattr(loaded, "predict")


class TestPrediction:
    @pytest.fixture
    def trained_model(self):
        df, _ = load_data()
        model, _ = train_model(df)
        return model

    def test_predict_setosa(self, trained_model):
        result = predict(trained_model, [5.1, 3.5, 1.4, 0.2])
        assert result["class_name"] == "setosa"
        assert 0 <= result["class_id"] <= 2

    def test_predict_returns_probabilities(self, trained_model):
        result = predict(trained_model, [6.7, 3.0, 5.2, 2.3])
        assert "probabilities" in result
        probs = result["probabilities"]
        assert len(probs) == 3
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_predict_invalid_input_shape(self, trained_model):
        with pytest.raises(ValueError):
            predict(trained_model, [1.0, 2.0])
