import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import ta
import plotly.graph_objects as go
from prophet import Prophet

st.title("Stock Price Tracker & Visualizer")
st.write("""
This is a stock price tracker for global markets.  
Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA) and select a time period to view historical stock data and moving averages.
""")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS)", "AAPL")
start_date, end_date = st.date_input(
    "Select Date Range",
    value=(datetime.date(2024, 1, 1), datetime.date.today())
)

short_window = st.slider("Short-Term MA (days)", min_value=2, max_value=50, value=10)
long_window = st.slider("Long-Term MA (days)", min_value=10, max_value=200, value=30)

if ticker:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            st.warning("⚠️ No data available. Try a different symbol or period.")
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            # Calculate MAs
            data = stock.history(start=start_date, end=end_date)
            data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
            data['Long_MA'] = data['Close'].rolling(window=long_window).mean()

            data['Signal'] = 0  # default: no signal
            data['Signal'][short_window:] = np.where(
                data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0
            )
            data['Position'] = data['Signal'].diff()

            # Show table
            st.subheader("Stock Data")
            st.dataframe(data)

            # Plot
            # Plot with Buy/Sell Signals
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(data['Close'], label="Closing Price", color='blue')
            ax.plot(data['Short_MA'], label=f"{short_window}-Day Short MA", color='orange')
            ax.plot(data['Long_MA'], label=f"{long_window}-Day Long MA", color='green')

            # Plot Buy Signals (Golden Cross)
            ax.plot(data[data['Position'] == 1].index,
                    data['Short_MA'][data['Position'] == 1],
                    '^', markersize=10, color='green', label='Buy Signal')

            # Plot Sell Signals (Death Cross)
            ax.plot(data[data['Position'] == -1].index,
                    data['Short_MA'][data['Position'] == -1],
                    'v', markersize=10, color='red', label='Sell Signal')

            ax.set_title("Stock Price with Moving Averages and Buy/Sell Signals")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)


            # Stock Analysis
            st.subheader("📈 Stock Analysis")
            highest_price = data['High'].max()
            lowest_price = data['Low'].min()
            pct_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100

            st.write(f"**Highest Price:** ${highest_price:.2f}")
            st.write(f"**Lowest Price:** ${lowest_price:.2f}")
            st.write(f"**Percentage Change:** {pct_change:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")

data = data.copy()
data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
data['MACD'] = ta.trend.MACD(data['Close']).macd()

fig_candle = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])
st.plotly_chart(fig_candle)


df_prophet = data[['Close']].reset_index()
df_prophet.columns = ['ds', 'y']

m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

st.write("### 30-Day Forecast")
fig2 = m.plot(forecast)

tickers = st.multiselect("Compare Stocks", ["AAPL", "MSFT", "GOOG", "TSLA"], default=["AAPL"])

for t in tickers:
    df = yf.download(t, start=start_date, end=end_date)['Close']
    st.line_chart(df.rename(t))


csv = data.to_csv().encode('utf-8')
st.download_button(
    "Download CSV",
    csv,
    "stock_data.csv",
    "text/csv",
    key='download-csv'
)

