
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request, jsonify
import backtrader as bt
import pyfolio as pf
from pyfolio.timeseries import perf_stats
from pyfolio.utils import extract_rets_pos_txn_from_cerebro
import ccxt

class Container:
    def __init__(self, xopt, transaction_costs, fees, taxes, market_liquidity, market_volatility, real_world_considerations, var, sharpe_ratio, metrics, model):
        self.xopt = xopt
        self.transaction_costs = transaction_costs
        self.fees = fees
        self.taxes = taxes
        self.market_liquidity = market_liquidity
        self.market_volatility = market_volatility
        self.real_world_considerations = real_world_considerations
        self.var = var
        self.sharpe_ratio = sharpe_ratio
        self.metrics = metrics
        self.model = model

def preprocess_data(data):
    try:
        data = data.dropna()
        data = data.pivot(columns='symbol', values=['return', 'volatility'])
        data.columns = data.columns.get_level_values(1)
        data = data.rename_axis(None, axis=1)
    except:
        raise ValueError("Error in preprocessing data")
    return data

def get_live_data(api_key, symbols):
    try:
        exchange = ccxt.binance({
            'rateLimit': 2000,
            'enableRateLimit': True,
            'verbose': True,
            'apiKey': api_key
        })
        timeframe = '1d'
        data = {}
        for symbol in symbols:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['return'] = df['close'].pct_change()
            df['volatility'] = df['return'].rolling(window=252).std() * np.sqrt(252)
            df['symbol'] = symbol
            data[symbol] = df
        data = pd.concat(data)
        data = data.reset_index(drop=True)
    except:
        raise ValueError("Error in retrieving live data")
    return data

def set_risk_tolerance(data, risk_tolerance):
    try:
        data['risk_tolerance'] = risk_tolerance
    except:
        raise ValueError("Error in setting risk tolerance")
    return data

def diversify_portfolio(data, sectors, industries, countries):
    try:
        data['sector'] = data['sector'].map(sectors)
        data['industry'] = data['industry'].map(industries)
        data['country'] = data['country'].map(countries)
    except:
        raise ValueError("Error in diversifying portfolio")
    return data

def calculate_transaction_costs(x, data, transaction_cost):
    try:
        transaction_costs = x.T @ data['volume'] * transaction_cost
    except:
        raise ValueError("Error in calculating transaction costs")
    return transaction_costs

def calculate_fees(x, data, management_fee, performance_fee):
    try:
        management_fees = x.T @ data['market_value'] * management_fee
        performance_fees = (x.T @ data['return'] - data['risk_free_rate'].mean()) * x.T @ data['market_value'] * performance_fee
    except:
        raise ValueError("Error in calculating fees")
    return management_fees, performance_fees

def calculate_taxes(x, data, tax_rate):
    try:
        taxes = x.T @ data['return'] * tax_rate
    except:
        raise ValueError("Error in calculating taxes")
    return taxes

def calculate_market_liquidity(x, data):
    try:
        market_liquidity = -x.T @ np.log(data['volume'])
    except:
        raise ValueError("Error in calculating market liquidity")
    return market_liquidity

def calculate_market_volatility(x, data):
    try:
        market_volatility = x.T @ data['volatility']
    except:
        raise ValueError("Error in calculating market volatility")
    return market_volatility

def calculate_real_world_considerations(x, data, transaction_costs, fees, taxes, market_liquidity, market_volatility):
    try:
        real_world_considerations = transaction_costs + sum(fees) + taxes + market_liquidity + market_volatility
    except:
        raise ValueError("Error in calculating real world considerations")
    return real_world_considerations

def calculate_var(data, x, initial_investment, confidence_level):
    try:
        portfolio_returns = x.T @ data['return']
        var = initial_investment - (initial_investment * (1 + portfolio_returns).cumprod()).quantile(confidence_level)
    except:
        raise ValueError("Error in calculating Value at Risk (VaR)")
    return var

def calculate_sharpe_ratio(data, x, risk_free_rate):
    try:
        portfolio_returns = x.T @ data['return']
        sharpe_ratio = (portfolio_returns - risk_free_rate) / (x.T @ data['volatility'])
    except:
        raise ValueError("Error in calculating Sharpe ratio")
    return sharpe_ratio

def backtest_portfolio(data, x):
    try:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(bt.Strategy)
        for i in range(data.shape[1]):
            cerebro.adddata(data.iloc[:, i])
        cerebro.run()
        results = extract_rets_pos_txn_from_cerebro(cerebro)
    except:
        raise ValueError("Error in backtesting portfolio")
    return results

def performance_evaluation(results):
    try:
        metrics = perf_stats(results)
    except:
        raise ValueError("Error in performance evaluation")
    return metrics

def train_ml_model(data, target):
    try:
        X = data.drop(target, axis=1)
        y = data[target]
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
        model = RandomForestRegressor()
        model.fit(X, y)
    except:
        raise ValueError("Error in training machine learning model")
    return model

def optimize_portfolio(data, initial_investment, transaction_cost, management_fee, performance_fee, tax_rate, risk_free_rate, risk_tolerance, confidence_level):
    try:
        data = preprocess_data(get_live_data(api_key, symbols))
        data = set_risk_tolerance(data, risk_tolerance)
        data = diversify_portfolio(data, sectors, industries, countries)
        bounds = [(0, 1) for i in range(data.shape[1])]
        x0 = [1/data.shape[1] for i in range(data.shape[1])]
        def objective_function(x, data, initial_investment, transaction_cost, management_fee, performance_fee, tax_rate, risk_free_rate):
            transaction_costs = calculate_transaction_costs(x, data, transaction_cost)
            management_fees, performance_fees = calculate_fees(x, data, management_fee, performance_fee)
            taxes = calculate_taxes(x, data, tax_rate)
            market_liquidity = calculate_market_liquidity(x, data)
            market_volatility = calculate_market_volatility(x, data)
            real_world_considerations = calculate_real_world_considerations(x, data, transaction_costs, (management_fees, performance_fees), taxes, market_liquidity, market_volatility)
            var = calculate_var(data, x, initial_investment, confidence_level)
            sharpe_ratio = calculate_sharpe_ratio(data, x, risk_free_rate)
            return -sharpe_ratio
        const = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x})
        result = minimize(objective_function, x0, args=(data, initial_investment, transaction_cost, management_fee, performance_fee, tax_rate, risk_free_rate), bounds=bounds, constraints=const)
        xopt = result.x
        transaction_costs = calculate_transaction_costs(xopt, data, transaction_cost)
        management_fees, performance_fees = calculate_fees(xopt, data, management_fee, performance_fee)
        taxes = calculate_taxes(xopt, data, tax_rate)
        market_liquidity = calculate_market_liquidity(xopt, data)
        market_volatility = calculate_market_volatility(xopt, data)
        real_world_considerations = calculate_real_world_considerations(xopt, data, transaction_costs, (management_fees, performance_fees), taxes, market_liquidity, market_volatility)
        var = calculate_var(data, xopt, initial_investment, confidence_level)
        sharpe_ratio = calculate_sharpe_ratio(data, xopt, risk_free_rate)
        results = backtest_portfolio(data, xopt)
        metrics = performance_evaluation(results)
        model = train_ml_model(data, 'return')
        container = Container(xopt, transaction_costs, (management_fees, performance_fees), taxes, market_liquidity, market_volatility, real_world_considerations, var, sharpe_ratio, metrics, model)
    except:
        raise ValueError("Error in optimizing portfolio")
    return container

def execute_trade(container, api_key):
    try:
        exchange = ccxt.binance({
            'rateLimit': 2000,
            'enableRateLimit': True,
            'verbose': True,
            'apiKey': api_key
        })
        for i, symbol in enumerate(data.columns):
            quantity = container.xopt[i] * initial_investment / data.iloc[-1, i]
            order = exchange.create_order(symbol, type='market', side='buy', amount=quantity)
    except:
        raise ValueError("Error in executing trade")
    return "Trades executed successfully"

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        api_key = request.form['api_key']
        symbols = request.form.getlist('symbols')
        initial_investment = float(request.form['initial_investment'])
        transaction_cost = float(request.form['transaction_cost'])
        management_fee = float(request.form['management_fee'])
        performance_fee = float(request.form['performance_fee'])
        tax_rate = float(request.form['tax_rate'])
        risk_free_rate = float(request.form['risk_free_rate'])
        risk_tolerance = float(request.form['risk_tolerance'])
        confidence_level = float(request.form['confidence_level'])
        sectors = request.form.getlist('sectors')
        industries = request.form.getlist('industries')
        countries = request.form.getlist('countries')
        container = optimize_portfolio(data, initial_investment, transaction_cost, management_fee, performance_fee, tax_rate, risk_free_rate, risk_tolerance, confidence_level)
        return render_template('results.html', container=container)
    return render_template('index.html')

@app.route('/execute_trade', methods=['POST'])
def execute_trade_route():
    result = execute_trade(container, api_key)
    return result

if __name__ == '__main__':
    app.run(debug=True)




