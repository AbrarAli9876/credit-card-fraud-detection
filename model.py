import joblib
import pandas as pd
import os
import numpy as np

# --- Define Base Directory for Model Loading ---
# This ensures model.pkl is expected in the SAME directory as this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --- End Base Directory Definition ---

MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# Check if the model file exists before attempting to load
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file '{MODEL_PATH}' not found. "
                            "Please ensure 'creditcard.csv' is in the same directory as 'train_model.py' "
                            "and then run 'python train_model.py' to generate the model in this directory.")

# Load the trained model and scaler
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    print(f"Model and scaler loaded successfully from {MODEL_PATH}.")

    # Get the exact feature names and their types from the loaded model
    if hasattr(model, 'feature_names_in_'):
        # Convert to plain Python strings for consistent internal use
        EXPECTED_MODEL_FEATURE_NAMES = [str(col) for col in model.feature_names_in_]
        # Also store the original types if needed for input DataFrame construction
        FEATURE_NAME_TYPES = {str(col): type(col) for col in model.feature_names_in_}
        print(f"Model expects features (from loaded model): {list(model.feature_names_in_)}")
    else:
        # Fallback (less likely with current train_model.py)
        # Assume standard strings for safety if feature_names_in_ is absent
        EXPECTED_MODEL_FEATURE_NAMES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        FEATURE_NAME_TYPES = {col: str for col in EXPECTED_MODEL_FEATURE_NAMES}
        print("Warning: Loaded model does not have 'feature_names_in_' attribute. Falling back to default feature names.")

except Exception as e:
    raise RuntimeError(f"Error loading model or scaler from '{MODEL_PATH}': {e}")


def predict_fraud(transaction_features: dict) -> tuple[int, float]:
    """
    Predicts whether a credit card transaction is fraudulent using the loaded XGBoost model.

    Args:
        transaction_features (dict): A dictionary containing the transaction features.
                                     Keys should match the feature names used during training
                                     (e.g., 'Time', 'V1', ..., 'V28', 'Amount').

    Returns:
        tuple[int, float]: A tuple containing:
                           - Prediction (0 for not fraud, 1 for fraud)
                           - Probability of being fraud
    """
    # Create the dictionary with keys converted to the expected type (str or np.str_)
    processed_transaction_features = {
        FEATURE_NAME_TYPES.get(key, str)(key): value
        for key, value in transaction_features.items()
    }
    input_df = pd.DataFrame([processed_transaction_features])

    # Ensure all expected columns from the model are present in the input DataFrame
    for col_name in EXPECTED_MODEL_FEATURE_NAMES:
        if col_name not in input_df.columns:
            # Handle cases where the UI might not provide all V-features (e.g., if expander is closed)
            # Ensure the missing column is added with the correct type (e.g., np.str_)
            input_df[FEATURE_NAME_TYPES.get(col_name, str)(col_name)] = 0.0

    # Reorder columns to exactly match the order the model was trained on.
    # This is crucial for correct prediction.
    input_df = input_df[EXPECTED_MODEL_FEATURE_NAMES]

    # Convert the DataFrame columns to match the type expected by the model (e.g., all str or all np.str_)
    if all(isinstance(f, np.str_) for f in model.feature_names_in_):
        input_df.columns = [np.str_(col) for col in input_df.columns]
    elif all(isinstance(f, str) for f in model.feature_names_in_):
         input_df.columns = [str(col) for col in input_df.columns]


    print(f"Features DataFrame columns *before scaling* (sent to model.py for prediction): {input_df.columns.tolist()}")

    # Apply the same scaling used during training to 'Time' and 'Amount'.
    df_scaled = input_df.copy()
    
    # Use the expected column names and types for scaling
    # Ensure keys used here match the types in df_scaled's columns
    df_scaled[FEATURE_NAME_TYPES['Time']('Time')] = scaler.transform(df_scaled[[FEATURE_NAME_TYPES['Time']('Time')]])
    df_scaled[FEATURE_NAME_TYPES['Amount']('Amount')] = scaler.transform(df_scaled[[FEATURE_NAME_TYPES['Amount']('Amount')]])

    print(f"Features DataFrame columns *after scaling* (sent to model for prediction): {df_scaled.columns.tolist()}")
    print(f"Features DataFrame head (after scaling): \n{df_scaled.head()}")

    # Get the prediction (0 or 1)
    prediction = model.predict(df_scaled)[0]
    # Get the probability of the transaction being fraudulent (class 1)
    prediction_proba = model.predict_proba(df_scaled)[:, 1][0]

    return int(prediction), float(prediction_proba)
