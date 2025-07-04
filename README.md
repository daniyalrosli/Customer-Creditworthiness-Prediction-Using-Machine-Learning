# Customer Creditworthiness Prediction Using Machine Learning

## Overview
This project provides a machine learning solution to predict a customer's creditworthiness based on financial behavior, transaction history, and demographic features. The model assists financial institutions in assessing lending risk and making informed credit decisions.

## Features
- **Data Preprocessing:** Handles missing values, encodes categorical variables, and normalizes data.
- **Feature Engineering:** Creates new features from transaction history and aggregates key financial metrics.
- **Imbalanced Data Handling:** Uses SMOTE to address class imbalance.
- **Model Building:** Trains a Random Forest classifier (and other algorithms) to classify customers as creditworthy or not.
- **Model Evaluation:** Assesses performance using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and feature importance.
- **Hyperparameter Tuning:** Optimizes models using GridSearchCV.
- **Visualization & EDA:** Provides exploratory data analysis and visualizations to understand trends and relationships.
- **Interactive Dashboard:** Streamlit app for data exploration, model evaluation, and making predictions.

## Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - Data Manipulation: `pandas`, `numpy`
  - Data Visualization: `matplotlib`, `seaborn`, `plotly`
  - Machine Learning: `scikit-learn`, `imblearn` (SMOTE)
  - Model Deployment: `streamlit`

## Project Structure
- `credit.py` — Main Streamlit dashboard for interactive analysis and prediction
- `credit.ipynb` — Jupyter notebook for exploratory data analysis and model development
- `credit_risk_dataset.csv` — Main dataset
- `dataset/` — Additional data or resources

## Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Customer-Creditworthiness-Prediction-Using-Machine-Learning
   ```
2. **(Recommended) Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn plotly streamlit
   ```

## Usage
### 1. Run the Streamlit Dashboard
```bash
streamlit run credit.py
```
- Navigate through the dashboard to explore data, analyze features, evaluate model performance, and make predictions.

### 2. Run the Jupyter Notebook
Open `credit.ipynb` in JupyterLab or VSCode to review the exploratory data analysis and model development process.

## Example
- Predict customer creditworthiness by uploading or using the provided dataset.
- Visualize feature importance and model performance metrics interactively.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## License
See `dataset/license.txt` for dataset license. Project code is provided under the MIT License (add a LICENSE file if needed).
