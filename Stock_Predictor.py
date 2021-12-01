# Import Libraries
from altair.vegalite.v4.schema.core import Day
import streamlit as st
from datetime import date
import datetime as dt
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# Set up dates
Start_Date = '2015-01-01'
Today = date.today().strftime('%Y-%m-%d')
Current = date.today()
Day_1 = date.today() + dt.timedelta(days=1)
Day_1


# Title
st.title("Stock Prediction App")

# List of stocks

stocks_dict = {'Please select':'', '1nvest S&P500':'ETF5IT.JO', 'African Rainbow Minerals Limited':'ARI.JO', 'Apple':'AAPL', 'ARK Innovation ETF': 'ARKK', 'Discovery':'DSY.JO', 'EOH Holdings Limited':'EOH.JO',
                                     'Google':'GOOG', 'Microsoft':'MSFT', 'Tongaat Hulett Limited': 'TON.JO'}
stocks = ('Please select', '1nvest S&P500', 'African Rainbow Minerals Limited', 'Apple', 'ARK Innovation ETF', 'Discovery', 'EOH Holdings Limited', 'Google', 'Microsoft', 'Tongaat Hulett Limited')
selected_stock = st.selectbox("Select stock for prediction", stocks)

st.write('Stock selected:', stocks_dict[selected_stock])

try:
    # Number of years predicting
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    # Function to get data
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, Start_Date, Today)
        data.reset_index(inplace=True)
        return data

    # Loading text
    data_load_state = st.text("Load data...")
    data = load_data(stocks_dict[selected_stock])
    data_load_state.text("Loading data...done!")

    # Raw data table
    st.subheader('Raw data')
    st.table(data.tail(10))

    # Plot raw data graph
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
        fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_data()

    # Forecasting
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={'Date':'ds' , 'Close' : 'y'})

    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    # Forecast data table
    st.subheader('Forecast data')
    forecast_new = forecast
    forecast_new = forecast_new.rename(columns={'ds':'Date', 'yhat':'Prediction'})
    forecast_new = forecast_new[['Date','Prediction']]
    st.write(forecast_new.tail(10))

    # Plot forecast data graph
    st.write('Forecast Data')
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    st.write('Forecast Components')
    fig2 = model.plot_components(forecast)
    st.write(fig2)
except:
    st.write('Please select a stock')