import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
import yfinance as yf
from datetime import datetime

# Initialize the Dash app
app = dash.Dash(__name__)

# Function to get historical stock data
def get_stock_data(stock_symbol, start_date, end_date):
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

# Forecasting function using Linear Regression
def forecast_stock(data, forecast_days=30):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = np.array(data['Days']).reshape(-1, 1)
    y = data['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_days = np.array(range(data['Days'].max() + 1, data['Days'].max() + 1 + forecast_days)).reshape(-1, 1)
    future_prices = model.predict(future_days)
    
    forecast_df = pd.DataFrame({'Date': pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=forecast_days, freq='D'),
                                'Forecast': future_prices})
    return forecast_df

# Calculate moving averages
def calculate_moving_averages(data, window=20):
    data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
    return data

# Define the layout of the app
app.layout = html.Div([
    html.H1(children='STOCK-VIZ-FORECAST', style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'color': 'black'}),
    
    html.Label('Select Stock Symbol:', style={'color': 'black'}),
    dcc.Dropdown(
        id='stock-dropdown',
        options=[
            {'label': 'Apple (AAPL)', 'value': 'AAPL'},
            {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
            {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
            {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
            {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
            {'label': 'Netflix (NFLX)', 'value': 'NFLX'},
            {'label': 'NVIDIA (NVDA)', 'value': 'NVDA'},
            {'label': 'Facebook (META)', 'value': 'META'},
            {'label': 'Adobe (ADBE)', 'value': 'ADBE'},
            {'label': 'Intel (INTC)', 'value': 'INTC'},
            {'label': 'PayPal (PYPL)', 'value': 'PYPL'},
            {'label': 'Shopify (SHOP)', 'value': 'SHOP'},
            {'label': 'Square (SQ)', 'value': 'SQ'},
            {'label': 'Twitter (TWTR)', 'value': 'TWTR'},
            {'label': 'Zoom (ZM)', 'value': 'ZM'}
        ],
        value='AAPL'
    ),
    
    html.Label('Select Date Range:', style={'color': 'black'}),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date='2020-01-01',
        end_date=datetime.today().strftime('%Y-%m-%d'),
        display_format='YYYY-MM-DD',
        style={'color': 'black'}
    ),
    
    html.Div([
        html.Label('Select Forecast Period:', style={'color': 'black'}),
        dcc.Dropdown(
            id='forecast-period-dropdown',
            options=[
                {'label': '7 Days', 'value': 7},
                {'label': '14 Days', 'value': 14},
                {'label': '30 Days', 'value': 30},
                {'label': '60 Days', 'value': 60},
                {'label': '90 Days', 'value': 90}
            ],
            value=30  
        ),
    ], style={'margin-top': '10px'}),
    
    html.Label('Select Chart:', style={'color': 'black'}),
    dcc.Dropdown(
        id='chart-type-dropdown',
        options=[
            {'label': 'Line Chart', 'value': 'line'},
            {'label': 'Candlestick Chart', 'value': 'candlestick'},
            {'label': 'Volume Chart', 'value': 'volume'},
            {'label': 'Moving Average', 'value': 'moving-average'},
            {'label': 'Bar Chart', 'value': 'bar'},
            {'label': 'Scatter Plot', 'value': 'scatter'}
        ],
        value='line' 
    ),
    
    dcc.Graph(id='selected-chart', style={'overflowX': 'scroll'}),

   
    dcc.Interval(
        id='interval-component',
        interval=1*60000,  # in milliseconds
        n_intervals=0
    )
], style={'backgroundColor': 'white'})

# Callback to update selected chart based on user's choice
@app.callback(
    Output('selected-chart', 'figure'),
    [Input('chart-type-dropdown', 'value'),
     Input('stock-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('forecast-period-dropdown', 'value')]
)
def update_chart(chart_type, stock_symbol, start_date, end_date, forecast_period):
    try:
        # Get historical stock data
        stock_data = get_stock_data(stock_symbol, start_date, end_date)
        if stock_data.empty:
            return go.Figure()
        
        # Get forecasted data
        forecast_data = forecast_stock(stock_data, forecast_days=forecast_period)
        
        if chart_type == 'line':
            trace = go.Scatter(
                x=stock_data['Date'],
                y=stock_data['Close'],
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue')
            )

            forecast_trace = go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast'],
                mode='lines',
                name='Forecasted Prices',
                line=dict(color='black', dash='dash')
            )

            layout = go.Layout(
                title=f'Stock Price for {stock_symbol}',
                xaxis={'title': 'Date', 'rangeslider': {'visible': True}},
                yaxis={'title': 'Price'},
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': 'black'}
            )

            return {'data': [trace, forecast_trace], 'layout': layout}
        
        elif chart_type == 'candlestick':
            trace = go.Candlestick(
                x=stock_data['Date'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick Chart'
            )

            forecast_trace = go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast'],
                mode='lines',
                name='Forecasted Prices',
                line=dict(color='black', dash='dash')
            )

            layout = go.Layout(
                title=f'Candlestick Chart for {stock_symbol}',
                xaxis={'title': 'Date', 'rangeslider': {'visible': True}},
                yaxis={'title': 'Price'},
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': 'black'}
            )

            return {'data': [trace, forecast_trace], 'layout': layout}
        
        elif chart_type == 'volume':
            trace = go.Bar(
                x=stock_data['Date'],
                y=stock_data['Volume'],
                name='Volume',
                marker=dict(color='orange')
            )

            layout = go.Layout(
                title=f'Stock Volume for {stock_symbol}',
                xaxis={'title': 'Date', 'rangeslider': {'visible': True}},
                yaxis={'title': 'Volume'},
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': 'black'}
            )

            return {'data': [trace], 'layout': layout}
        
        elif chart_type == 'moving-average':
            stock_data = calculate_moving_averages(stock_data)
            
            trace = go.Scatter(
                x=stock_data['Date'],
                y=stock_data['MA20'],
                mode='lines',
                name='20-Day Moving Average',
                line=dict(color='green')
            )

            forecast_trace = go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast'],
                mode='lines',
                name='Forecasted Prices',
                line=dict(color='black', dash='dash')
            )

            layout = go.Layout(
                title=f'Moving Average for {stock_symbol}',
                xaxis={'title': 'Date', 'rangeslider': {'visible': True}},
                yaxis={'title': 'Price'},
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': 'black'}
            )

            return {'data': [trace, forecast_trace], 'layout': layout}
        
        elif chart_type == 'bar':
            trace = go.Bar(
                x=stock_data['Date'],
                y=stock_data['Close'],
                name='Close Price',
                marker=dict(color='purple')
            )

            forecast_trace = go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast'],
                mode='lines',
                name='Forecasted Prices',
                line=dict(color='black', dash='dash')
            )

            layout = go.Layout(
                title=f'Bar Chart for {stock_symbol}',
                xaxis={'title': 'Date', 'rangeslider': {'visible': True}},
                yaxis={'title': 'Close Price'},
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': 'black'}
            )

            return {'data': [trace, forecast_trace], 'layout': layout}
        
        elif chart_type == 'scatter':
            trace = go.Scatter(
                x=stock_data['Date'],
                y=stock_data['Close'],
                mode='markers',
                name='Close Price',
                marker=dict(color='red')
            )

            forecast_trace = go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast'],
                mode='lines',
                name='Forecasted Prices',
                line=dict(color='black', dash='dash')
            )

            layout = go.Layout(
                title=f'Scatter Plot for {stock_symbol}',
                xaxis={'title': 'Date', 'rangeslider': {'visible': True}},
                yaxis={'title': 'Close Price'},
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': 'black'}
            )

            return {'data': [trace, forecast_trace], 'layout': layout}
    
    except Exception as e:
        print(f"Error updating chart: {e}")
        return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
