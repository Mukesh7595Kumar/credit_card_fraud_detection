# Credit Card Fraud Detection Web Application

This project is a Flask-based web application that uses machine learning models to detect fraudulent credit card transactions. Users can enter transaction data and choose between an LSTM model and an XGBoost model to predict if the transaction is fraudulent or legitimate.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Workflow](#workflow)
- [Contributing](#contributing)
- [License](#license)

## Overview
Credit card fraud detection is essential to protect financial transactions. This application uses two machine learning models—LSTM and XGBoost—to classify transactions based on the input data. The web interface enables users to enter transaction details and choose a model for fraud detection.

## Project Structure
- `app.py`: Main Flask application file.
- `templates/`: Contains HTML templates for the homepage (`home.html`) and result page (`result.html`).
- `static/`: Contains static files like CSS for styling.
- `models/`: Holds the pre-trained models:
  - `Fraud_Detection_Model.keras` (LSTM model)
  - `xgb_bal_smote_model.bin` (XGBoost model)

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Mukesh7595Kumar/credit_card_fraud_detection.git
    cd credit_card_fraud_detection
    ```

2. **Set up a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3. **Prepare the Models**:
   - Download or save the pre-trained LSTM model as `Fraud_Detection_Model.keras` and the XGBoost model as `xgb_bal_smote_model.bin` in the `models` directory.

## Usage

1. **Run the Application**:
    ```bash
    python app.py
    ```
   This starts the application locally at `http://127.0.0.1:5000`.

2. **Access the Web Interface**:
   - Go to `http://127.0.0.1:5000` in your browser.
   - Enter transaction details in the form, separating values with spaces.
   - Select the model (LSTM or XGBoost) and click "Submit" to get the prediction.

## Models

- **LSTM Model**:
  - **File**: `models/Fraud_Detection_Model.keras`
  - **Description**: An LSTM neural network model, designed for sequential data, to detect fraudulent transactions.
  
- **XGBoost Model**:
  - **File**: `models/xgb_bal_smote_model.bin`
  - **Description**: XGBoost classifier, optimized for structured, imbalanced data, effective in fraud detection.

## Workflow

1. **Homepage**: Users enter transaction data and choose a model on the main page (`home.html`).
2. **Prediction Route (`/predict`)**:
   - Captures user input, processes it into a format suitable for the selected model.
   - Predicts the outcome (fraud or legitimate) based on the input.
3. **Results Page**: Displays the prediction result on `result.html`.

## Contributing

To contribute:
1. Fork this repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make and commit your changes (`git commit -am 'Add new feature'`).
4. Push the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License.
