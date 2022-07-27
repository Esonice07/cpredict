import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas_datareader as web

df = pd.DataFrame()

crypto_symbols = ['BTC', 'ETH', 'LTC', 'DOGE', 'SOL', 'USDT', 'USDC', 'BNB', 'XRP', 'ADA', 'DAI', 'WTRX', 'DOT', 'LEVER', 'ACH', 'HEX', 'TRX',
                 'SHIB', 'LEO', 'WBTC', 'AVAX', 'YOUC', 'DESO', 'DAI', 'LINK', 'XLM',  'MATIC', 'UNI1', 'STETH', 'LTC', 'FTT']


def load_data_yahoo(symbol):
    return web.Datareader(symbol, 'yahoo', dt.datatime(2016, 1, 1,), dt.datatime.now())


for coins in crypto_symbols:
    coin_index = crypto_symbols.index(coins)

    #yf_data = yf.Ticker(f'{coins}-USD').history(start='2014-01-01', end=dt.datetime.now(), interval='1d')
    yf_data = load_data_yahoo(f'{coins}-USD')
    cdf = pd.DataFrame(yf_data)
    cdf['symbol'] = coins
    df = df.append(cdf)

df.to_csv('yf_cryptodata.csv')
df.reset_index(inplace=True)
df.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)

selected_stocks = st.selectbox("Select dataset for prediction", crypto_symbols)

#n_years = int(input('Enter the number of years: '))
n_years = st.slider("Years of prediction:", 1, 7)
period = n_years * 365

data_load_state = st.text('Load data...')
data = df[df.symbol==selected_stocks]
data_load_state.text('Loading data...done!')

st.subheader('Statistical Summary of Selected Coin')
st.write(data.describe())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Frecasting

df_train = data[['Date', 'Close']]

df_train = df_train.rename(columns={'Date':"ds", "Close": 'y'})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)


