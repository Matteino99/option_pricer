from flask import Flask, render_template, request
import math
from scipy.stats import norm

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
    return render_template('index_complex2.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    spot_price = float(request.form['spot_price'])
    strike = float(request.form['strike'])
    time_to_expiration = float(request.form['time_to_expiration'])
    volatility = float(request.form['volatility'])
    interest_rate = float(request.form['interest_rate'])
    dividend_yield = float(request.form['dividend_yield'])
    option_type = str(request.form['option_type'])
    
    # Call your options pricing function here
    options_price = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type)

    return render_template('index_complex2.html', options_price=options_price)

if __name__ == '__main__':
    app.run(debug=False, port=8003)
