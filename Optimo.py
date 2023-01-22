

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from flask import Flask, render_template, request
import os
import pyfolio as pf
import backtrader as bt

def preprocess_data(data):
    try:
        data = data.dropna()
        data = data.drop_duplicates()
        data = data.sort_values(by='Date')
    except:
        raise ValueError("Error in preprocessing data")
    return data

def set_risk_tolerance(data, risk_tolerance):
    try:
        data['return'] = data['Close'].pct_change()
        data['rolling_mean'] = data['return'].rolling(window=30).mean()
        data['scaled_return'] = data['return'] / data['rolling_mean'] * risk_tolerance
    except:
        raise ValueError("Error in setting risk tolerance")
    return data

def diversify_portfolio(data, sectors, industries, countries):
    try:
        data = data[data['Sector'].isin(sectors)]
        data = data[data['Industry'].isin(industries)]
        data = data[data['Country'].isin(countries)]
    except:
        raise ValueError("Error in diversifying portfolio")
    return data

def optimize_portfolio(data, initial_investment):
    try:
        n = data.shape[0]
        x0 = np.ones(n) / n
        bounds = [(0, 1) for i in range(n)]
        def portfolio_return(x, data):
            return -np.dot(x, data['scaled_return'])
        def portfolio_variance(x, data):
            return np.dot(x, np.dot(data['return'].cov(), x))
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        options = {'disp': True}
        res = minimize(portfolio_variance, x0, args=(data,), bounds=bounds, constraints=constraint, options=options)
        xopt = res.x
        log = res.fun
    except:
        raise ValueError("Error in optimizing portfolio")
    return xopt, log

def calculate_transaction_costs(xopt, data, transaction_cost):
    try:
        transaction_costs = transaction_cost * np.abs(np.diff(xopt) / xopt[:-1]).sum()
    except:
        raise ValueError("Error in calculating transaction costs")
    return transaction_costs

def calculate_fees(xopt, data, management_fee, performance_fee):
    try:
        management_fees = management_fee * xopt.sum()
        performance_fees = performance_fee * (np.dot(xopt, data['return']) - management_fees)
    except:
        raise ValueError("Error in calculating management and performance fees")
    return management_fees, performance_fees

def calculate_taxes(xopt, data, tax_rate):
    try:
        taxes = tax_rate * np.dot(xopt, data['return'])
    except:
        raise ValueError("Error in calculating taxes")
    return taxes

def calculate_market_liquidity(xopt, data):
    try:
        market_liquidity = np.dot(xopt, data['Market Capitalization'])
    except:
        raise ValueError("Error in calculating market liquidity")
    return market_liquidity

def calculate_market_volatility(xopt, data):
    try:
        market_volatility = np.dot(xopt, data['Volatility'])
    except:
        raise ValueError("Error in calculating market volatility")
    return market_volatility

def calculate_real_world_considerations(xopt, data, transaction_costs, fees, taxes, market_liquidity, market_volatility):
    try:
        real_world_considerations = transaction_costs + fees[0] + fees[1] + taxes + market_liquidity + market_volatility
    except:
        raise ValueError("Error in calculating real-world considerations")
    return real_world_considerations

def calculate_var(data, xopt, initial_investment, confidence_level):
    try:
            var = initial_investment - initial_investment * (1 - confidence_level)
    except:
        raise ValueError("Error in calculating Value at Risk")
    return var

def calculate_sharpe_ratio(data, xopt, initial_investment, risk_free_rate):
    try:
        sharpe_ratio = (np.dot(xopt, data['return']) - risk_free_rate) / (np.dot(xopt, data['return'].std()))
    except:
        raise ValueError("Error in calculating Sharpe Ratio")
    return sharpe_ratio

def backtest_portfolio(xopt, data, initial_investment):
    try:
        portfolio_value = initial_investment * (1 + np.dot(xopt, data['return']))
        # Implement backtesting library here
    except:
        raise ValueError("Error in backtesting portfolio")
    return portfolio_value

def performance_evaluation(portfolio_value):
    try:
        # Implement evaluation metrics such as return, risk and sharpe ratio
        # using libraries such as pyfolio and backtrader
    except:
        raise ValueError("Error in evaluating portfolio performance")
    return metrics

def train_ml_model(data, target_col):
    try:
        scaler = RobustScaler()
        data[data.columns] = scaler.fit_transform(data[data.columns])
        x = data.drop(target_col, axis=1)
        y = data[target_col]
        model = RandomForestRegressor()
        model.fit(x, y)
    except:
        raise ValueError("Error in training machine learning model")
    return model

def get_live_data(api_key, symbols):
    try:
        data = yf.download(symbols, api_key=api_key)
    except:
        raise ValueError("Error in retrieving live data")
    return data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        api_key = os.environ.get('API_KEY')
        symbols = request.form['symbols']
        risk_tolerance = float(request.form['risk_tolerance'])
        initial_investment = float(request.form['initial_investment'])
        transaction_cost = float(request.form['transaction_cost'])
        management_fee = float(request.form['management_fee'])
        performance_fee = float(request.form['performance_fee'])
        tax_rate = float(request.form['tax_rate'])
        sectors = request.form.getlist('sectors')
        industries = request.form.getlist('industries')
        countries = request.form.getlist('countries')
        data = preprocess_data(get_live_data(api_key, symbols))
        data = set_risk_tolerance(data, risk_tolerance)
        data = diversify_portfolio(data, sectors, industries, countries)
        x0 = np.ones(len(data)) / len(data)
        bounds = [(0, 1) for i in range(len(data))]
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        res = minimize(lambda x: -np.dot(x, data['return']), x0, bounds=bounds, constraints=constraints, method='SLSQP')
        xopt = res.x
        transaction_costs = calculate_transaction_costs(xopt, data, transaction_cost)
        fees = calculate_fees(xopt, data, management_fee, performance_fee)
        taxes = calculate_taxes(xopt, data, tax_rate)
        market_liquidity = calculate_market_liquidity(xopt, data)
        market_volatility = calculate_market_volatility(xopt, data)
        real_world_considerations = calculate_real_world_considerations(xopt, data, transaction_costs, fees, taxes, market_liquidity, market_volatility)
        var = calculate_var(data, xopt, initial_investment, 0.99)
        sharpe_ratio = calculate_sharpe_ratio(data, xopt, initial_investment, 0.03)
        portfolio_value = backtest_portfolio(xopt, data, initial_investment)
        metrics = performance_evaluation(portfolio_value)
        model = train_ml_model(data, 'return')
        container = Container(xopt, transaction_costs, fees, taxes, market_liquidity, market_volatility, real_world_considerations, var, sharpe_ratio, metrics, model)
        return jsonify(container.__dict__)
    except:
        return jsonify({'error': 'An error occurred while optimizing the portfolio'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


