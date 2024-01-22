import math
import numpy as np

def BinomialEuropeanOption(spot_price, strike, interest_rate, time_to_expiration, num_steps, up_factor, down_factor, dividend_yield, volatility, option_type ):
    
    """
Calculate the price of a European option using the binomial tree model.

Parameters:
- spot_price (float): Current stock price.
- strike (float): Option strike price.
- interest_rate (float): Annual risk-free interest rate.
- time_to_expiration (float): Time to option expiration in years.
- num_steps (int): Number of time steps in the binomial tree.
- up_factor (float): Factor by which the stock price increases in the up state.
- down_factor (float): Factor by which the stock price decreases in the down state.
- dividend_yield (float): Annual dividend yield.
- volatility (float): Annual volatility of the stock.
- option_type (str): "call" for call option, "put" for put option.

Returns:
- option_price (float): Calculated option price.
- terminal_stock_prices (numpy.ndarray): Array of terminal stock prices.
"""
    
    num_terminal_nodes = num_steps + 1  # Number of terminal nodes of tree
    u = 1 + up_factor  # Expected value in the up state
    d = 1 - down_factor  # Expected value in the down state
    dt = time_to_expiration / float(num_steps)
    df = math.exp(- (interest_rate - dividend_yield) * dt)
    qu = (math.exp((interest_rate - dividend_yield) * dt) - d) / (u - d)
    qd = 1 - qu
    STs = np.zeros(num_terminal_nodes)

    for i in range(num_terminal_nodes):
        STs[i] = spot_price * (u**(num_steps - i)) * (d**i)

    if option_type == "call":
        payoffs = np.maximum(0, STs - strike)
    else:
        payoffs = np.maximum(0, strike - STs)

    for i in range(num_steps):
        payoffs = (payoffs[:-1] * qu + payoffs[1:] * qd) * df

    return payoffs[0]

def binomial_american_option(spot_price, strike, interest_rate, time_to_expiration, num_steps, up_factor, down_factor, dividend_yield, volatility, option_type):
    
    """
Calculate the price of an American option using the binomial tree model.

Parameters:
- spot_price (float): Current stock price.
- strike (float): Option strike price.
- interest_rate (float): Annual risk-free interest rate.
- time_to_expiration (float): Time to option expiration in years.
- num_steps (int): Number of time steps in the binomial tree.
- up_factor (float): Factor by which the stock price increases in the up state.
- down_factor (float): Factor by which the stock price decreases in the down state.
- dividend_yield (float): Annual dividend yield.
- volatility (float): Annual volatility of the stock.
- option_type (str): "call" for call option, "put" for put option.

Returns:
- option_price (float): Calculated option price.
- terminal_stock_prices (numpy.ndarray): Array of terminal stock prices.
"""
    
    u = 1 + up_factor  # Expected value in the up state
    d = 1 - down_factor  # Expected value in the down state
    dt = time_to_expiration / float(num_steps)
    df = np.exp(-interest_rate * dt)
    qu = (np.exp((interest_rate - dividend_yield) * dt) - d) / (u - d)
    qd = 1 - qu

    # Initialize a 2D tree at T=0
    STs = [np.array([spot_price])]

    # Simulate the possible stock prices path
    for i in range(num_steps):
        prev_branches = STs[-1]
        st = np.concatenate((prev_branches * u, [prev_branches[-1] * d]))
        STs.append(st)  # Add nodes at each time step

        if option_type == "call":
            payoffs = np.maximum(0, STs[num_steps] - strike)
        else:
            payoffs = np.maximum(0, strike - STs[num_steps])
 
    # Calculate option price using vectorized calculations
    for i in reversed(range(num_steps)):
        # The payoffs from NOT exercising the option
        payoffs = (payoffs[:-1] * qu + payoffs[1:] * qd) * df
        # Payoffs from exercising, for American options
        if option_type == "call":
            payoffs = np.maximum(payoffs, STs[i] - strike)
        else:
            payoffs = np.maximum(payoffs, strike - STs[i])

    return payoffs[0]

# Example usage:
spot_price = 50
strike = 52
interest_rate = 0.05
time_to_expiration = 2
num_steps = 2
up_factor = 0.2
down_factor = 0.2
dividend_yield = 0.0
volatility = 0.3
option_type = "call"  # Change to "call" for a call option

#result = binomial_american_option(spot_price, strike, interest_rate, time_to_expiration, num_steps, up_factor, down_factor, dividend_yield, volatility, option_type)
#print(f"Option price is: {result}")
result = BinomialEuropeanOption(spot_price, strike, interest_rate, time_to_expiration, num_steps, up_factor, down_factor, dividend_yield, volatility, option_type)
print(f"Option price is: {result}")