"""Generate competition-format predictions from a saved pipeline."""
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from src.config import MODEL_FILE, TEST_FILE, PREDICTION_FILE
from src.data_validation import validate_prediction_data
from src.preprocessing import prepare_features, transform_absence_categories

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def predict(input_path=TEST_FILE, output_path=PREDICTION_FILE, model_path=MODEL_FILE):
    model = joblib.load(model_path)
    frame = pd.read_csv(input_path)
    validate_prediction_data(frame)
    features = transform_absence_categories(prepare_features(frame))
    predictions = np.expm1(model.predict(features))
    predictions = np.maximum(predictions, 1.0)
    output = pd.DataFrame({"Id": frame["Id"].to_numpy(), "SalePrice": predictions})
    if len(output) != len(frame) or not np.isfinite(output["SalePrice"]).all() or (output["SalePrice"] <= 0).any():
        raise ValueError("Prediction validation failed.")
    output_path = __import__("pathlib").Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    logging.info("Wrote %d predictions to %s", len(output), output_path)
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(TEST_FILE), help="Input CSV path")
    parser.add_argument("--output", default=str(PREDICTION_FILE), help="Output CSV path")
    parser.add_argument("--model", default=str(MODEL_FILE), help="Saved joblib pipeline path")
    args=parser.parse_args(); predict(args.input,args.output,args.model)

if __name__ == "__main__": main()
