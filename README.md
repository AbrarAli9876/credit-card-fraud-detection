💳 Credit Card Fraud Detection App
This project implements a web-based application for detecting fraudulent credit card transactions using a Machine Learning model. The application provides an interactive interface for users to input transaction details and receive an instant prediction on whether the transaction is legitimate or fraudulent.

✨ Features
Interactive Input: User-friendly interface to input Transaction Time, Transaction Amount, and anonymized V1-V28 features.

Machine Learning Prediction: Utilizes a pre-trained XGBoost Classifier to predict the likelihood of fraud.

Probability Output: Displays the predicted class (Legitimate/Fraudulent) and the associated probability of fraud.

Real-time Feedback: Provides immediate visual feedback on the prediction result.

Responsive Design: Built with Streamlit for a clean and adaptive user interface.

🚀 Technologies Used
Python 3.10

Streamlit: For building the interactive web application.

Pandas: For data manipulation.

Scikit-learn: For data preprocessing (StandardScaler) and general machine learning utilities.

XGBoost: The powerful gradient boosting library used for the classification model.

Imbalanced-learn (SMOTE): For handling class imbalance in the dataset during model training.

Joblib: For saving and loading the trained machine learning model and scaler.

🛠️ Setup and Installation
Follow these steps to get your application up and running locally.

1. Clone the Repository (or create project folder)
First, ensure you have a project directory setup. If this is a new project, create a folder like CREDIT CARD FRAUD DETECTION APP.

2. Download the Dataset
The model is trained on a specific dataset.

Download creditcard.csv from Kaggle: Credit Card Fraud Detection Dataset

Place the downloaded creditcard.csv file directly into your project directory (e.g., CREDIT CARD FRAUD DETECTION APP/creditcard.csv).

3. Set Up Your Python Environment
It's highly recommended to use a virtual environment to manage dependencies. You can use either venv (standard Python) or conda.

Option A: Using venv (Standard Python)
Open your terminal or command prompt.

Navigate into your project directory:

cd "C:\Users\ksaab\OneDrive\Documents\Python\CREDIT CARD FRAUD DETECTION APP\"


Create a virtual environment:

python -m venv venv 


Activate the virtual environment:

On Windows:

.\venv\Scripts\activate


On macOS/Linux:

source venv/bin/activate


Option B: Using conda (Recommended if you have Anaconda/Miniconda)
Open your terminal or Anaconda Prompt.

Navigate into your project directory:

cd "C:\Users\ksaab\OneDrive\Documents\Python\CREDIT CARD FRAUD DETECTION APP\"


Create a Conda environment with Python 3.10:

conda create -p .\venv python=3.10 -y


Activate the environment:

conda activate .\venv


4. Install Dependencies
Once your virtual environment is active, install the required packages using pip:

pip install -r requirements.txt


(If you don't have a requirements.txt file yet, create one in your project directory with the following content and then run the command above:)

# requirements.txt
pandas
scikit-learn
xgboost
imbalanced-learn
joblib
streamlit


5. Train the Machine Learning Model
The application requires a pre-trained model. This script will train the model and save it as model.pkl in your project directory.

Ensure your virtual environment is activated (from step 3).

Run the training script from your project directory:

python train_model.py


Important: This step will print information about the model training process, including performance metrics and confirmation that model.pkl has been saved. Ensure there are no errors here.

6. Run the Streamlit Application
Finally, launch the web application:

Ensure your virtual environment is activated.

Run the Streamlit app from your project directory:

streamlit run app.py


This will open the application in your default web browser (usually at http://localhost:8501).

👨‍💻 How to Use the Application
Input Transaction Details:

Transaction Time: Enter the time in minutes since the first transaction in the dataset.

Transaction Amount: Enter the monetary value of the transaction (in ₹).

Anonymized Features (V1-V28): You can expand the "Adjust V Features (Advanced)" section to manually tweak these values. By default, they are set to 0.0. While not intuitively understandable, these features are crucial for the model's prediction.

Analyze Transaction: Click the "🚀 Analyze Transaction" button to get the model's prediction.

View Results: The application will display:

A clear message indicating if the transaction is FRAUDULENT or LIKELY LEGITIMATE.

The Probability of Fraud, showing the model's confidence level.

⚠️ Important Note
This application is designed for demonstration and educational purposes only. A robust, production-grade fraud detection system would typically involve:

More complex models and ensemble techniques.

Continuous data pipeline integration.

Advanced alert mechanisms.

Human-in-the-loop validation and ongoing model monitoring.

Handling of real-world data intricacies and evolving fraud patterns.

👤 Developed By
Name: K.S. Abrar Ali Ahmed

Degree Pursuing: Artificial Intelligence and Data Science

College Name: K S School Of Engineering and Management

📄 License
This project is open-source and available under the MIT License. You are free to modify and distribute the code for personal and educational purposes.