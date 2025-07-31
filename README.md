# 🚗 Car Price Predictor
A machine learning project that predicts the selling price of used cars based on various features like brand, manufacturing year, kilometers driven, fuel type, and more. This project demonstrates end-to-end implementation of data cleaning, feature engineering, model building, and evaluation.

# 📁 Project Structure
bash
Copy
Edit
car-price-predictor/
├── data/                  # Raw and cleaned datasets
├── notebooks/             # Jupyter notebooks for EDA and modeling
├── scripts/               # Python scripts for data processing and modeling
├── models/                # Trained ML model files (e.g., .pkl, .joblib)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

# 📊 Dataset Overview
Source: Fetching from Quikr.com (used car listings)
Target Variable: Price (INR)

# Features Used:
name
year
kms_driven
fuel_type
owner_type
location

# 🧹 Data Preprocessing
Removed rows with missing or invalid values (e.g., "Ask For Price")
Extracted brand from car name
Converted string to numeric (e.g., kms_driven, year)
Encoded categorical variables using Label and One-Hot Encoding
Removed outliers using IQR and z-score methods

# 📈 Exploratory Data Analysis
Visualizations were used to explore:
Price distribution across brands
Fuel type vs price
Kilometers driven vs price
Correlation heatmaps

# 🤖 Model Training
Models Used:
Linear Regression

Evaluation Metrics:
R² Score

# 💻 Technologies Used
Tool / Library	Purpose
Python	Core programming language
Pandas	Data manipulation
NumPy	Numerical operations
Matplotlib & Seaborn	Data visualization
Scikit-learn	Machine learning models & evaluation
Jupyter Notebook	Development environment

# ⚠️ Problems Faced
❗ Dirty Data: Many listings had inconsistent formats and missing values like "Ask For Price" or "NA".

❗ Unstructured Text: Car names were unstructured, requiring parsing to extract brand names.

❗ Outliers: Extreme values in kilometers driven and price skewed model performance until treated.

❗ Multicollinearity: Some features were highly correlated, affecting regression model assumptions.
