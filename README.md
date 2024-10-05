# Credit-Scoring-Model
# Credit Scoring Model

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Modeling](#modeling)
- [Visualization](#visualization)
- [License](#license)

## Project Overview
This project aims to develop a **Credit Scoring Model** for Bati Bank, focusing on implementing a buy-now-pay-later service in collaboration with an eCommerce company. The model classifies users into high or low-risk categories based on their transaction data.

## Data Description
The dataset contains the following columns:
- **TransactionId**: Unique identifier for each transaction
- **BatchId**: Identifier for transaction batches
- **AccountId**: Identifier for customer accounts
- **SubscriptionId**: Identifier for subscriptions
- **CustomerId**: Unique customer identifier
- **CurrencyCode**: Code for the currency used
- **CountryCode**: Country of the transaction
- **ProviderId**: Identifier for the service provider
- **ProductId**: Identifier for the purchased product
- **Amount**: Transaction amount (numeric)
- **Value**: Transaction value (numeric)
- **TransactionStartTime**: Timestamp of the transaction
- **PricingStrategy**: Pricing strategy applied
- **FraudResult**: Result of fraud detection
- **Total_Transaction_Amount**: Total amount for transactions
- **Transaction_Count**: Number of transactions
- **Product Categories**: Various product categories (categorical)
- **ChannelId**: Identifiers for channels

## Installation
To get started with this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/credit-scoring-model.git
    ```

2. Navigate to the project directory:
    ```bash
    cd credit-scoring-model
    ```

3. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

4. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
After installing the required packages, you can run the following script to execute the model:

bash
  python main.py

## Features
- **Data Preprocessing**: Handles missing values, encodes categorical variables, and splits the dataset.
- **Model Training**: Implements machine learning algorithms to classify risk.
- **Model Evaluation**: Evaluates model performance using appropriate metrics.
- **Visualization**: Provides visualizations of risk categories and transaction trends.

## Modeling
This project employs various machine learning techniques, including:
- **Logistic Regression**
- **Decision Trees**
- **Random Forest**
- **Gradient Boosting**

## Visualization
Visualizations are created using **Matplotlib** and **Seaborn** to illustrate the distribution of risk categories and other insights from the dataset.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
