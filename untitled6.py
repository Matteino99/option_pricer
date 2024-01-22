from flask import Flask, render_template, request
import numpy as np
import math
import os
from scipy.stats import norm
#from waitress import serve
from tutte_le_opzioni import calculate_options_price, binomial_european_option, binomial_american_option, barrier_option_pricer, digital_pricer

app = Flask(__name__, template_folder='templates')

def get_option_categories():
    return ["European_BS", "European_binomial", "American_binomial", "Barrier", "Digital"]

@app.route('/')
def index():
    return render_template('select num_type.html')


@app.route('/select_options_input', methods=['POST'])
def select_options_type():
    num_options = int(request.form['num_options'])
    option_category = request.form['option_category']
    
    if option_category == "European_BS":
        return render_template('form_european_BS.html', num_options=num_options, option_category=option_category)
    elif option_category == "European_binomial":
        return render_template('form_european_binomial.html', num_options=num_options, option_category=option_category)    
    elif option_category == "American_binomial":
        return render_template('form_american_binomial.html', num_options=num_options, option_category=option_category)
    elif option_category == "Barrier":
        return render_template('form_barrier.html', num_options=num_options, option_category=option_category)
    elif option_category == "Digital":
        return render_template('form_digital.html', num_options=num_options, option_category=option_category)
    else:
        return "Invalid option category"



@app.route('/calculate', methods=['POST'])
def calculate():
    total_price = 0
    total_delta = 0    
    total_gamma = 0
    total_vega = 0
    total_theta = 0
    total_rho = 0
    num_options = int(request.form['num_options'])
    option_category = request.form['option_category']

    
    for i in range(1, num_options + 1):
        spot_price = float(request.form[f'spot_price{i}'])
        strike = float(request.form[f'strike{i}'])
        time_to_expiration = float(request.form[f'time_to_expiration{i}'])
        volatility = float(request.form[f'volatility{i}'])
        interest_rate = float(request.form[f'interest_rate{i}'])
        dividend_yield = float(request.form[f'dividend_yield{i}'])
        option_type = request.form[f'option_type{i}']
    
        if option_category == "European_BS":
            price, delta, gamma, vega, theta, rho  = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type)           
        elif option_category == "European_binomial":
            num_steps = int(request.form[f'num_steps{i}'])
            up_factor = float(request.form[f'up_factor{i}'])
            down_factor = float(request.form[f'down_factor{i}'])
            price = binomial_european_option(spot_price, strike, interest_rate, time_to_expiration, num_steps, up_factor, down_factor, dividend_yield, option_type)
            delta, gamma, vega, theta, rho = 0, 0, 0, 0, 0  # Set Greeks to 0 as they are not needed
        elif option_category == "American_binomial":
            num_steps = int(request.form[f'num_steps{i}'])
            up_factor = float(request.form[f'up_factor{i}'])
            down_factor = float(request.form[f'down_factor{i}'])
            price = binomial_american_option(spot_price, strike, interest_rate, time_to_expiration, num_steps, up_factor, down_factor, dividend_yield, option_type)
            delta, gamma, vega, theta, rho = 0, 0, 0, 0, 0
        elif option_category == "Barrier":
            barrier = float(request.form[f'barrier{i}'])
            rebate = float(request.form[f'rebate{i}'])
            barrier_type = request.form[f'barrier_type{i}']
            observation = float(request.form[f'observation{i}'])
            price = barrier_option_pricer(spot_price, strike, barrier, rebate, time_to_expiration, volatility, interest_rate, dividend_yield, option_type, barrier_type, observation)
            delta, gamma, vega, theta, rho = 0, 0, 0, 0, 0
        elif option_category == "Digital":
            coupon = float(request.form[f'coupon{i}'])
            barrier_shift = float(request.form[f'barrier_shift{i}'])
            options_style = request.form[f'options_style{i}']
            price = digital_pricer(spot_price, strike, coupon, time_to_expiration, volatility, interest_rate, dividend_yield, barrier_shift, option_type, options_style)[0]
            delta, gamma, vega, theta, rho = 0, 0, 0, 0, 0
        else:
            return "Invalid option category"

        total_price += price
        total_delta += delta
        total_gamma += gamma
        total_vega += vega
        total_theta += theta
        total_rho += rho
        
    return render_template('result2.html', total_price=round(total_price,4), total_delta=round(total_delta,4), total_gamma=round(total_gamma,4), total_vega=round(total_vega,4), total_theta=round(total_theta,4), total_rho=round(total_rho,4))

if __name__ == '__main__':
    app.run(debug=True, port=5556)

serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5556)))
