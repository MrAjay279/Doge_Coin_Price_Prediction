import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(
    page_title="Dogecoin Price Prediction",
    page_icon="🚀🐶🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    body { background-color: #1e1e1e; color: white; }
    .block-container { padding: 1rem 2rem; background-color: #2a2a2a; border-radius: 10px; }
    h1, h2, h3 { color: #ffcc00; }
    .stTextInput > label { font-size: 1.1rem; color: #ffcc00; }
    .stTable { background-color: #3a3a3a; }
    .stTabs { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title
st.title('🚀🐶🐕 Dogecoin Price Prediction')

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("dogecoin_data.csv")
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data = data.sort_values(by='Date')
    data.set_index('Date', inplace=True)
    return data

data = load_data()

def convert_volume(vol):
    if 'M' in vol:
        return float(vol.replace('M', '')) * 1e6
    elif 'B' in vol:
        return float(vol.replace('B', '')) * 1e9
    return float(vol)

data['Vol.'] = data['Vol.'].apply(convert_volume)
data['Change %'] = data['Change %'].str.replace('%', '').astype(float)

data['Prev_Price'] = data['Price'].shift(1)
data.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Vol.', 'Prev_Price', 'Change %']
X = data[features]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("Mean Squared Error", f"${mse:.6f}")
col2.metric("Mean Absolute Error", f"${mae:.6f}")
col3.metric("R-squared", f"${r2:.6f}")

def predict_future_prices(model, last_known_values, num_days, feature_names):
    predictions = []
    for _ in range(num_days):
        feature_values = pd.DataFrame([last_known_values], columns=feature_names)
        predicted_price = model.predict(feature_values)[0]
        predictions.append(predicted_price)
        last_known_values = np.append(last_known_values[1:], predicted_price)
    return predictions

st.sidebar.header('User Input')
num_days_input = st.sidebar.text_input("Enter number of days to predict:", "10")
try:
    num_days = int(num_days_input)
    if num_days < 1 or num_days > 1000:
        st.sidebar.error("Please enter a number between 1 and 1000.")
        num_days = 10
except ValueError:
    st.sidebar.error("Invalid input! Please enter a valid integer.")
    num_days = 10

last_known_values = data.iloc[-1][features].values
future_prices = predict_future_prices(model, last_known_values, num_days, features)

st.subheader('Predicted Prices')
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, num_days + 1)]
predicted_df = pd.DataFrame({"Date": future_dates, "Predicted_Price": [f"${price:.6f}" for price in future_prices]})
st.dataframe(predicted_df.style.set_properties(**{'background-color': '#3a3a3a', 'color': 'white'}))

st.subheader('Visualization')
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['📈 Historical Prices', '🔮 Predicted Prices', '📊 Correlation Matrix', '📉 Pairplot', '✅ Actual vs Predicted', '🔥 Feature Importance'])

with tab1:
    st.line_chart(data['Price'])
with tab2:
    st.line_chart(predicted_df.set_index('Date'))
with tab3:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
with tab4:
    fig = sns.pairplot(data[features + ['Price']])
    st.pyplot(fig)
with tab5:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, color='cyan')
    ax.set_xlabel('Actual Prices')
    ax.set_ylabel('Predicted Prices')
    ax.set_title('Actual vs Predicted Prices')
    st.pyplot(fig)
with tab6:
    feature_importance = model.feature_importances_
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=features, y=feature_importance, ax=ax, palette='viridis')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)
