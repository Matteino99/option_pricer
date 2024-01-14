from flask import Flask, render_template, request
import math
import os
from scipy.stats import norm
from waitress import serve



app = Flask(__name__, template_folder='templates')

def calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type):
    # Placeholder Black-Scholes formula - replace with your actual pricing model
    # This is a basic example and may not be suitable for all cases
    # You might want to use a library like `numpy` or `scipy` for more complex calculations

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
        options_price = spot_price * math.exp(-dividend_yield * time_to_expiration) * norm.cdf(d1) - strike * math.exp(-interest_rate * time_to_expiration) * norm.cdf(d2)
    else:
        options_price = strike * math.exp(-interest_rate * time_to_expiration) * norm.cdf(-d2) - spot_price * math.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1) 

    print("Options Price:", options_price)

    return options_price

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
    num_options = int(request.form['num_options'])


    for i in range(1, num_options + 1):
        spot_price = float(request.form[f'spot_price{i}'])
        strike = float(request.form[f'strike{i}'])
        time_to_expiration = float(request.form[f'time_to_expiration{i}'])
        volatility = float(request.form[f'volatility{i}'])
        interest_rate = float(request.form[f'interest_rate{i}'])
        dividend_yield = float(request.form[f'dividend_yield{i}'])
        option_type = request.form[f'option_type{i}']

        options_price = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type)
        total_price += options_price

    return render_template('result.html', total_price=round(total_price,4))



if __name__ == '__main__':
    app.run(debug=True, port=8004)

serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8004)))
