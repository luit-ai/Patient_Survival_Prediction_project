"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.metrics import mean_squared_error, r2_score

from patient_survival_prediction_model.predict import make_prediction

import numpy as np


def test_make_prediction(sample_input_data):
    # When
    result = make_prediction(input_data=sample_input_data[0])
    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list), "Predictions should be a array of strings message"
    for prediction in predictions:
        assert "The patient is predicted" in prediction, "Prediction message should indicate patient status"
    assert result.get("errors") is None, "There should be no errors in the result"
    #assert len([predictions]) == expected_num_of_predictions, f"Expected {expected_num_of_predictions} predictions, got {len([predictions])}"
    assert result.get("version") is not None, "Version should be present in the result"

