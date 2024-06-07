import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# Set page config
st.set_page_config(page_title="Stock Forecast App", page_icon=":chart_with_upwards_trend:", layout="wide")



# Initialize session state
if 'load_state' not in st.session_state:
    st.session_state['load_state'] = False

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")



# Title and introduction text
st.title('StockSage: A webapp for predicting stock prices :chart_with_upwards_trend:')
st.markdown("""
This webapp predicts the stock price using a machine learning model called *Prophet*.
Select the desired stock and the forecast period, and see the magic happen!
""")

# Expanded list of stocks
stock_options = ['GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 'FB', 'TSLA', 'BRK.A', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'DIS', 'NVDA', 'HD', 'PYPL', 'BAC', 'VZ', 'ADBE', 'CMCSA', 'NFLX', 'KO', 'NKE', 'MRK', 'PEP', 'T', 'PFE', 'INTC']  

# Sidebar settings
st.sidebar.header('User Input Parameters')
selected_stock = st.sidebar.selectbox('Select dataset for prediction', stock_options)
user_input = st.sidebar.text_input("Or enter a stock ticker here:", '')

# Use the user input if provided, else use the selected option
selected_stock = user_input.upper() if user_input else selected_stock

n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365


# Using st.expander to organize the selection and cache info
with st.sidebar.expander("ℹ️ - About this app", expanded=True):
    st.write("""
        - This is a project for Sophomore Seminar class at Fisk University
        - Navaraj Panday and Mahesh Yadav
        - This is a demonstration only and not financial advice.
    """)

    if st.button('Clear cache'):
        st.cache_data.clear()  # Clear the cache
        st.session_state['load_state'] = False

# Load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load state text
if not st.session_state['load_state']:
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')
    st.session_state['load_state'] = True
else:
    data = load_data(selected_stock)

# Container for the main content
with st.container():
    left_column, right_column = st.columns([3, 1])
    
    with left_column:
        st.subheader('Raw data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
        plot_raw_data()

    with right_column:
        st.subheader('Forecast controls')
        st.write('Use the slider to adjust the forecast period.')

# Predict forecast with Prophet
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})


m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Forecast components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)
