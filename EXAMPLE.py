# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 21:47:05 2024

@author: matte
"""

from flask import Flask, render_template, request
import math
from scipy.stats import norm

app = Flask(__name__, template_folder='templates')

def calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, option_type):
    # Placeholder Black-Scholes formula - replace with your actual pricing model
    # This is a basic example and may not be suitable for all cases
    # You might want to use a library like `numpy` or `scipy` for more complex calculations

    d1 = (math.log(spot_price / strike) + (interest_rate + (volatility ** 2) / 2) * time_to_expiration) / (volatility * math.sqrt(time_to_expiration))
    d2 = d1 - volatility * math.sqrt(time_to_expiration)
    
    if option_type == "call":
        options_price = spot_price * norm.cdf(d1) - strike * math.exp(-interest_rate * time_to_expiration) * norm.cdf(d2)
    else:
        options_price = strike * math.exp(-interest_rate * time_to_expiration) * norm.cdf(-d2) - spot_price * norm.cdf(-d1) 
    return options_price

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    spot_price = float(request.form['spot_price'])
    strike = float(request.form['strike'])
    time_to_expiration = float(request.form['time_to_expiration'])
    volatility = float(request.form['volatility'])
    interest_rate = float(request.form['interest_rate'])
    option_type = str(request.form['option_type'])
    
    # Call your options pricing function here
    options_price = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, option_type)

    return render_template('index.html', options_price=options_price)

if __name__ == '__main__':
    app.run(debug=False, port=8001)
       