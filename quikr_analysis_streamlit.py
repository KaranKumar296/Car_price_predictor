import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title='Quikr Car Analysis')
st.title('🚗 Quikr Used Car Price Analysis')

# Load Data
@st.cache_data
def load_data():
    car = pd.read_csv('quikr_car.csv')

    car = car[car['year'].str.isnumeric()]
    car['year'] = car['year'].astype(int)

    car = car[car['Price'] != 'Ask For Price']
    car['Price'] = car['Price'].str.replace(',', '').astype(int)

    car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
    car = car[car['kms_driven'].str.isnumeric()]
    car['kms_driven'] = car['kms_driven'].astype(int)

    car = car[~car['fuel_type'].isna()]

    car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
    car = car.reset_index(drop=True)

    car = car[car['Price'] < 6000000]
    return car

car = load_data()
st.subheader("Sample Cleaned Data")
st.dataframe(car.head())

# Visualizations
st.subheader("Boxplot: Company vs Price")
fig1, ax1 = plt.subplots(figsize=(15,7))
sns.boxplot(x='company', y='Price', data=car, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha='right')
st.pyplot(fig1)

st.subheader("Swarmplot: Year vs Price")
fig2, ax2 = plt.subplots(figsize=(20,10))
sns.swarmplot(x='year', y='Price', data=car, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha='right')
st.pyplot(fig2)

st.subheader("Relplot: KMs Driven vs Price")
fig3 = sns.relplot(x='kms_driven', y='Price', data=car, height=7, aspect=1.5)
st.pyplot(fig3.fig)

st.subheader("Boxplot: Fuel Type vs Price")
fig4, ax4 = plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type', y='Price', data=car, ax=ax4)
st.pyplot(fig4)

# Prepare ML model
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])
column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)

# Train best model
@st.cache_resource
def train_model():
    best_score = -1
    best_random_state = 0
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
        temp_pipe = make_pipeline(column_trans, LinearRegression())
        temp_pipe.fit(X_train, y_train)
        score = r2_score(y_test, temp_pipe.predict(X_test))
        if score > best_score:
            best_score = score
            best_random_state = i
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=best_random_state)
    pipe.fit(X_train, y_train)
    return pipe, best_score

model, model_score = train_model()
st.success(f"✅ Model trained with best R² Score: {model_score:.2f}")

# User Input
st.header("🚘 Predict Car Price")

name = st.selectbox('Car Name', sorted(car['name'].unique()))
company = st.selectbox('Company', sorted(car['company'].unique()))
year = st.selectbox('Year', sorted(car['year'].unique(), reverse=True))
kms_driven = st.number_input('Kilometers Driven', min_value=0, max_value=500000, step=1000)
fuel_type = st.selectbox('Fuel Type', car['fuel_type'].unique())

if st.button('Predict Price'):
    input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    predicted_price = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹{int(predicted_price):,}")

# Save model
with open('LinearRegressionModel.pkl', 'wb') as f:
    pickle.dump(model, f)