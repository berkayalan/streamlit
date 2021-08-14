from typing import get_origin
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from datetime import date
import yfinance as yf
from fbprophet import Prophet
#https://facebook.github.io/prophet/docs/installation.html
from fbprophet.plot import plot_plotly
#https://plotly.com/python/graph-objects/
from plotly import graph_objs as go 

#This is helpful resource: https://medium.com/analytics-vidhya/python-how-to-get-bitcoin-data-in-real-time-less-than-1-second-lag-38772da43740

start = "2018-01-01"
today= date.today().strftime("%Y-%m-%d")

@st.cache
def load_data(ticker):
    data= yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data 

# Plot raw data
def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],y=data["Open"],name="stock_open"))    
    fig.add_trace(go.Scatter(x=data["Date"],y=data["Close"],name="stock_close"))
    fig.layout.update(title_text="Time Series data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)    


st.title("Stock Price Prediction App")

#https://finance.yahoo.com/cryptocurrencies
stocks = ("WISH","AAPL","AMD","SOFI","AMC","PLTR","F","PFE","CLOV","ARCLK.IS","BPIRY","GOOG","MSFT","GME")
selected_stock=st.selectbox("Select Stock for prediction",stocks)

data_load_state= st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

st.subheader("Let's see current status!")
st.write(data.tail())

plot_raw_data()

st.subheader("Let's predict the future!")

n_years = st.slider("Years of Prediction",1,5)
period = n_years*365

df_predict= data["Date","Close"]
df_predict= df_predict.rename(columns={"Date":"ds","Close":"y"})

model = Prophet()
model.fit(df_predict)
future_prediction = model.make_future_dataframe(periods=period)
forecast = model.predict(future_prediction)

st.subheader("Let's see forecast results!")
st.write(forecast.tail())

st.write("Forecast Data")
fig1=plot_plotly(model,forecast)
st.plot_plotly(fig1)

st.write("Forecast Components")
fig2=model.plot_components(forecast)
st.write(fig2)
