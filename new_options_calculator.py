from flask import Flask, render_template, request
import numpy as np
import math
import os
from scipy.stats import norm
from waitress import serve


app = Flask(__name__, template_folder='templates')

def calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type):

    print("Spot Price:", spot_price)
    print("Strike:", strike)
    print("Time to Expiration:", time_to_expiration)
    print("Volatility:", volatility)
    print("Interest Rate:", interest_rate)
    print("Dividend Yield:", dividend_yield)
    print("Option Type:", option_type)
  
    d1 = (math.log(spot_price / strike) + (interest_rate - dividend_yield + (volatility ** 2) / 2) * time_to_expiration) / (volatility * math.sqrt(time_to_expiration))
    d2 = d1 - volatility * math.sqrt(time_to_expiration)
    
    if option_type == "call":
        price = spot_price * math.exp(-dividend_yield * time_to_expiration) * norm.cdf(d1) - strike * math.exp(-interest_rate * time_to_expiration) * norm.cdf(d2)
        delta = norm.cdf(d1, 0, 1)
        gamma = norm.pdf(d1, 0, 1) / (spot_price * volatility * np.sqrt(time_to_expiration))
        vega = spot_price * norm.pdf(d1, 0, 1) * np.sqrt(time_to_expiration) * 0.01 #vega for 1% change in volatility
        theta = (-spot_price * norm.pdf(d1, 0, 1) * volatility / (2 * np.sqrt(time_to_expiration)) - interest_rate * strike *np.exp(-interest_rate * time_to_expiration) * norm.cdf(d2, 0, 1))/365 #theta in days
        rho = (strike * time_to_expiration * np.exp(-interest_rate * time_to_expiration) * norm.cdf(d2, 0, 1))*0.01
        
    else:
        price = strike * math.exp(-interest_rate * time_to_expiration) * norm.cdf(-d2) - spot_price * math.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1) 
        delta = norm.cdf(d1, 0, 1)
        gamma = norm.pdf(d1, 0, 1) / (spot_price * volatility * np.sqrt(time_to_expiration))
        vega = spot_price * norm.pdf(d1, 0, 1) * np.sqrt(time_to_expiration) * 0.01 #vega for 1% change in volatility
        theta = (-spot_price * norm.pdf(d1, 0, 1) * volatility / (2 * np.sqrt(time_to_expiration)) + interest_rate * strike *np.exp(-interest_rate * time_to_expiration) * norm.cdf(-d2, 0, 1))/365 #theta in days
        rho = (-strike * time_to_expiration * np.exp(-interest_rate * time_to_expiration) * norm.cdf(-d2, 0, 1))*0.01
        
    print("Options Price:", price)
    print("Options delta:", delta)
    print("Options gamma:", gamma)
    print("Options vega:", vega)
    print("Options theta:", theta)
    print("Options rho:", rho)
    
    return price, delta, gamma, vega, theta, rho


@app.route('/')
def index():
    return render_template('select_num_options.html')

@app.route('/select_options', methods=['POST'])
def select_options():
    num_options = int(request.form['num_options'])
    return render_template('options_input.html', num_options=num_options)

@app.route('/calculate', methods=['POST'])
def calculate():
    total_price = 0
    total_delta = 0    
    total_gamma = 0
    total_vega = 0
    total_theta = 0
    total_rho = 0
    num_options = int(request.form['num_options'])


    for i in range(1, num_options + 1):
        spot_price = float(request.form[f'spot_price{i}'])
        strike = float(request.form[f'strike{i}'])
        time_to_expiration = float(request.form[f'time_to_expiration{i}'])
        volatility = float(request.form[f'volatility{i}'])
        interest_rate = float(request.form[f'interest_rate{i}'])
        dividend_yield = float(request.form[f'dividend_yield{i}'])
        option_type = request.form[f'option_type{i}']

        price, delta, gamma, vega, theta, rho = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type)
        total_price += price
        total_delta += delta
        total_gamma += gamma
        total_vega += vega
        total_theta += theta
        total_rho += rho
        
        
    return render_template('result2.html', total_price=round(total_price,4), total_delta=round(total_delta,4), total_gamma=round(total_gamma,4), total_vega=round(total_vega,4), total_theta=round(total_theta,4), total_rho=round(total_rho,4))

if __name__ == '__main__':
    app.run(debug=True, port=8005)

serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8005)))
