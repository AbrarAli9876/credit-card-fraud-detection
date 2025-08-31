import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import joblib
import os
import numpy as np

# --- Define Base Directory for Model and Data ---
# This ensures model.pkl and creditcard.csv are expected in the SAME directory as this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --- End Base Directory Definition ---

# --- Data Source Information ---
# You can download the 'creditcard.csv' dataset from Kaggle:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Please place the downloaded 'creditcard.csv' file in the SAME directory as this script.

def train_and_save_model(data_filename='creditcard.csv', model_filename='model.pkl'):
    """
    Trains an XGBoost model for credit card fraud detection with SMOTE for imbalance handling,
    and saves the trained model and scaler to a .pkl file in the script's directory.

    Args:
        data_filename (str): Name of the credit card transactions CSV file.
        model_filename (str): Name of the file to save the trained model.
    """
    full_data_path = os.path.join(BASE_DIR, data_filename)
    full_model_path = os.path.join(BASE_DIR, model_filename)

    if not os.path.exists(full_data_path):
        print(f"Error: Dataset '{data_filename}' not found at: {full_data_path}")
        print("Please download 'creditcard.csv' from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"and place it in the same directory as '{os.path.basename(__file__)}'.")
        return

    print(f"Loading data from {full_data_path}...")
    try:
        df = pd.read_csv(full_data_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Data loaded successfully. Preparing for training...")

    # --- Convert 'Time' from seconds to minutes ---
    df['Time'] = df['Time'] / 60
    print("Converted 'Time' feature from seconds to minutes.")

    # Separate features (X) and target (y)
    # The 'Class' column is the target variable (0: not fraud, 1: fraud)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Scale 'Time' and 'Amount' features as they are not anonymized
    # The V1-V28 features are already pre-scaled via PCA, so they don't need further scaling.
    scaler = StandardScaler()
    X['Time'] = scaler.fit_transform(X[['Time']])
    X['Amount'] = scaler.fit_transform(X[['Amount']])

    # Split data into training and testing sets
    # Using stratify=y to maintain the same proportion of fraud cases in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Applying SMOTE to the training data to handle class imbalance...")
    # Apply SMOTE to the training data only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Original training set shape: {X_train.shape}, Fraud cases: {y_train.sum()}")
    print(f"Resampled training set shape: {X_train_res.shape}, Fraud cases: {y_train_res.sum()}")

    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
    )
    model.fit(X_train_res, y_train_res)

    print("Model training complete. Evaluating model performance on the original test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report (on original test set):")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix (on original test set):")
    print(confusion_matrix(y_test, y_pred))

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auprc = auc(recall, precision)
    print(f"\nArea Under the Precision-Recall Curve (AUPRC): {auprc:.4f}")

    joblib.dump({'model': model, 'scaler': scaler}, full_model_path)
    print(f"\nModel and scaler saved successfully to {full_model_path}")
    if hasattr(model, 'feature_names_in_'):
        print(f"Features used for training (from model.feature_names_in_): {list(model.feature_names_in_)}")
    else:
        print("Warning: Model does not have 'feature_names_in_' attribute. This might be an older model or a different type.")

    print("Training script finished. You can now run 'streamlit run app.py'")

if __name__ == "__main__":
    train_and_save_model()
