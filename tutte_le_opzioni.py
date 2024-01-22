import math
import numpy as np
from scipy.stats import norm


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
        price = spot_price * math.exp(-dividend_yield * time_to_expiration) * norm.cdf(d1, 0, 1) - strike * math.exp(-interest_rate * time_to_expiration) * norm.cdf(d2, 0, 1)
        delta = norm.cdf(d1, 0, 1)
        gamma = norm.pdf(d1, 0, 1) / (spot_price * volatility * np.sqrt(time_to_expiration))
        vega = spot_price * norm.pdf(d1, 0, 1) * np.sqrt(time_to_expiration) * 0.01 #vega for 1% change in volatility
        theta = (-spot_price * norm.pdf(d1, 0, 1) * volatility / (2 * np.sqrt(time_to_expiration)) - interest_rate * strike * np.exp(-interest_rate * time_to_expiration) * norm.cdf(d2, 0, 1))/256 #theta in days
        rho = (strike * time_to_expiration * np.exp(-interest_rate * time_to_expiration) * norm.cdf(d2, 0, 1)) * 0.01
        
    else:
        price = strike * math.exp(-interest_rate * time_to_expiration) * norm.cdf(-d2,0, 1) - spot_price * math.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1, 0, 1) 
        delta = norm.cdf(d1, 0, 1)
        gamma = norm.pdf(d1, 0, 1) / (spot_price * volatility * np.sqrt(time_to_expiration))
        vega = spot_price * norm.pdf(d1, 0, 1) * np.sqrt(time_to_expiration) * 0.01 #vega for 1% change in volatility
        theta = (-spot_price * norm.pdf(d1, 0, 1) * volatility / (2 * np.sqrt(time_to_expiration)) + interest_rate * strike * np.exp(-interest_rate * time_to_expiration) * norm.cdf(-d2, 0, 1))/256 #theta in days
        rho = (-strike * time_to_expiration * np.exp(-interest_rate * time_to_expiration) * norm.cdf(-d2, 0, 1)) * 0.01

    return price, delta, gamma, vega, theta, rho        




def digital_pricer(spot_price, strike, coupon, time_to_expiration, volatility, interest_rate, dividend_yield, barrier_shift, option_type, options_style):
    """
    Calculate digital option price and theoretical price.

    Parameters:
    - spot_price (float): Current spot price.
    - strike (float): Option strike price.
    - coupon (float): Coupon value.
    - time_to_expiration (float): Time to option expiration in years.
    - volatility (float): Annual volatility of the underlying asset.
    - interest_rate (float): Annual risk-free interest rate.
    - dividend_yield (float): Annual dividend yield.
    - barrier_shift (float): Shift in the barrier (if applicable).
    - option_type (str): "call" for call option, "put" for put option.
    - options_style (str): "european" or other style.

    Returns:
    - digital_price (float): Calculated digital option price.
    - theo_price (float): Theoretical option price.
    """
    d1 = (math.log(spot_price / strike) + (interest_rate - dividend_yield + (volatility ** 2) / 2) * time_to_expiration) / (volatility * math.sqrt(time_to_expiration))
    d2 = d1 - volatility * math.sqrt(time_to_expiration)

    if option_type == "call":
        spot1 = spot_price - coupon / 2 + barrier_shift 
        price1 = calculate_options_price(spot1, strike, time_to_expiration, volatility, interest_rate, dividend_yield, "call")[0]
        spot2 = spot_price + coupon / 2 + barrier_shift 
        price2 = calculate_options_price(spot2, strike, time_to_expiration, volatility, interest_rate, dividend_yield, "call")[0]

        if options_style == "european":
            digital_price = np.subtract(price2, price1)
            theo_price = coupon * norm.cdf(d2, 0, 1) * math.exp(-interest_rate * time_to_expiration)
        else:
            digital_price = np.subtract(price2, price1) * 2  # due to reflection principle
            theo_price = coupon * norm.cdf(d2, 0, 1) * math.exp(-interest_rate * time_to_expiration) * 2

    else:
        spot1 = spot_price + coupon / 2
        price1 = calculate_options_price(spot1, strike, time_to_expiration, volatility, interest_rate, dividend_yield, "put")[0]
        spot2 = spot_price - coupon / 2
        price2 = calculate_options_price(spot2, strike, time_to_expiration, volatility, interest_rate, dividend_yield, "put")[0]

        if options_style == "european":
            digital_price = np.subtract(price2, price1)
            theo_price = coupon * norm.cdf(-d1, 0, 1) * math.exp(-interest_rate * time_to_expiration)
        else:
            digital_price = np.subtract(price2, price1) * 2  # due to reflection principle
            theo_price = coupon * norm.cdf(-d1, 0, 1) * math.exp(-interest_rate * time_to_expiration) * 2

    return np.maximum(0,digital_price), np.maximum(0,theo_price)



#THIS CODE IS VALIDE ONLY FOR REVERSE BARRIER UP CALL AND DOWN PUT

def barrier_option_pricer(spot_price, strike, barrier, rebate, time_to_expiration, volatility, interest_rate, dividend_yield, option_type, barrier_type, observation ):
    """
    Calculate the price of continuous barrier options.

    Parameters:
    - spot_price (float): Current spot price.
    - strike (float): Option strike price.
    - barrier (float): Barrier level.
    - rebate (float): Rebate amount.
    - time_to_expiration (float): Time to option expiration in years.
    - volatility (float): Annual volatility of the underlying asset.
    - interest_rate (float): Annual risk-free interest rate.
    - dividend_yield (float): Annual dividend yield.
    - option_type (str): "call" for call option, "put" for put option.
    - barrier_type (str): "in" for in-barrier, "out" for out-barrier.
    - observation (int): Number of observations.

    Returns:
    - option_price (float): Calculated option price.
    """

    lambdaa = (interest_rate - dividend_yield + ((volatility**2) / 2)) / volatility**2
    y = (math.log((barrier**2) / (strike * spot_price)) / (volatility * math.sqrt(time_to_expiration))) + lambdaa * volatility *  math.sqrt(time_to_expiration)
    x1 = (math.log(strike / barrier) / (volatility * math.sqrt(time_to_expiration) ) ) + lambdaa * volatility *  math.sqrt(time_to_expiration)
    y1 = (math.log(barrier / spot_price) / (volatility * math.sqrt(time_to_expiration))) + lambdaa * volatility *  math.sqrt(time_to_expiration)

    if observation == 0:
        if option_type == "call":
            if barrier_type == "in":
                barrier_option_price = spot_price * norm.cdf(x1, 0, 1) * math.exp(-dividend_yield * time_to_expiration) - strike * math.exp(-interest_rate * time_to_expiration) \
                    * norm.cdf(x1 - volatility * math.sqrt(time_to_expiration), 0, 1) - spot_price * math.exp(-dividend_yield * time_to_expiration) * ((barrier / spot_price)**(2 * lambdaa)) \
                        * ((norm.cdf(-y, 0, 1) - norm.cdf(-y1, 0, 1))) + strike * math.exp(-interest_rate * time_to_expiration) * ((barrier / spot_price)**((2 * lambdaa) -2)) * \
                            ((norm.cdf(-y + volatility * math.sqrt(time_to_expiration), 0, 1) - norm.cdf(-y1 + volatility * math.sqrt(time_to_expiration), 0, 1))) #IN CALL
            else: 
                up_in_call_price = spot_price * norm.cdf(x1, 0, 1) * math.exp(-dividend_yield * time_to_expiration) - strike * math.exp(-interest_rate * time_to_expiration) \
                    * norm.cdf(x1 - volatility * math.sqrt(time_to_expiration), 0, 1) - spot_price * math.exp(-dividend_yield * time_to_expiration) * ((barrier / spot_price)**(2 * lambdaa)) \
                        * ((norm.cdf(-y, 0, 1) - norm.cdf(-y1, 0, 1))) + strike * math.exp(-interest_rate * time_to_expiration) * ((barrier / spot_price)**((2 * lambdaa) -2)) * \
                            ((norm.cdf(-y + volatility * math.sqrt(time_to_expiration), 0, 1) - norm.cdf(-y1 + volatility * math.sqrt(time_to_expiration), 0, 1))) #IN CALL
                barrier_option_price = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type)[0] - up_in_call_price #IN CALL PRICE
        else: 
            if barrier_type == "in":
                    barrier_option_price = -spot_price * norm.cdf(-x1, 0, 1) * math.exp(-dividend_yield * time_to_expiration) + strike * math.exp(-interest_rate * time_to_expiration) \
                        * norm.cdf(-x1 + volatility * math.sqrt(time_to_expiration), 0, 1) + spot_price * math.exp(-dividend_yield * time_to_expiration) * ((barrier / spot_price)**(2 * lambdaa)) \
                            * ((norm.cdf(y, 0, 1) - norm.cdf(y1, 0, 1))) - strike * math.exp(-interest_rate * time_to_expiration) * ((barrier / spot_price)**((2 * lambdaa) -2)) * \
                                ((norm.cdf(y - volatility * math.sqrt(time_to_expiration), 0, 1) - norm.cdf(y1 - volatility * math.sqrt(time_to_expiration), 0, 1))) 
            else:
                in_put_price = -spot_price * norm.cdf(-x1, 0, 1) * math.exp(-dividend_yield * time_to_expiration) + strike * math.exp(-interest_rate * time_to_expiration) \
                    * norm.cdf(-x1 + volatility * math.sqrt(time_to_expiration), 0, 1) + spot_price * math.exp(-dividend_yield * time_to_expiration) * ((barrier / spot_price)**(2 * lambdaa)) \
                        * ((norm.cdf(y, 0, 1) - norm.cdf(y1, 0, 1))) - strike * math.exp(-interest_rate * time_to_expiration) * ((barrier / spot_price)**((2 * lambdaa) -2)) * \
                             ((norm.cdf(y - volatility * math.sqrt(time_to_expiration), 0, 1) - norm.cdf(y1 - volatility * math.sqrt(time_to_expiration), 0, 1))) 
                barrier_option_price = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type)[0] - in_put_price #IN CALL PRICE           
    else:
        if option_type == "call":
            if barrier_type == "in":
                barrier2 = barrier * math.exp(-0.5826 * volatility * math.sqrt(time_to_expiration / observation)) #SPOSTO LA BARRIERA PIU VICINA ALL'AUMENTARE DELLE OSSERVAZIONI COSI COSTA DI PIU
                barrier_option_price = spot_price * norm.cdf(x1, 0, 1) * math.exp(-dividend_yield * time_to_expiration) - strike * math.exp(-interest_rate * time_to_expiration) \
                    * norm.cdf(x1 - volatility * math.sqrt(time_to_expiration), 0, 1) - spot_price * math.exp(-dividend_yield * time_to_expiration) * ((barrier2 / spot_price)**(2 * lambdaa)) \
                        * ((norm.cdf(-y, 0, 1) - norm.cdf(-y1, 0, 1))) + strike * math.exp(-interest_rate * time_to_expiration) * ((barrier2 / spot_price)**((2 * lambdaa) -2)) * \
                            ((norm.cdf(-y + volatility * math.sqrt(time_to_expiration), 0, 1) - norm.cdf(-y1 + volatility * math.sqrt(time_to_expiration), 0, 1)))           
            else:
                barrier2 = barrier * math.exp(0.5826 * volatility * math.sqrt(time_to_expiration / observation)) #SHORT BARRIER
                up_in_call_price = spot_price * norm.cdf(x1, 0, 1) * math.exp(-dividend_yield * time_to_expiration) - strike * math.exp(-interest_rate * time_to_expiration) \
                    * norm.cdf(x1 - volatility * math.sqrt(time_to_expiration), 0, 1) - spot_price * math.exp(-dividend_yield * time_to_expiration) * ((barrier2 / spot_price)**(2 * lambdaa)) \
                        * ((norm.cdf(-y, 0, 1) - norm.cdf(-y1, 0, 1))) + strike * math.exp(-interest_rate * time_to_expiration) * ((barrier2 / spot_price)**((2 * lambdaa) -2)) * \
                            ((norm.cdf(-y + volatility * math.sqrt(time_to_expiration), 0, 1) - norm.cdf(-y1 + volatility * math.sqrt(time_to_expiration), 0, 1)))
                barrier_option_price = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type)[0] - up_in_call_price #IN CALL PRICE                                
        else:
            if barrier_type == "in":
                barrier2 = barrier * math.exp(0.5826 * volatility * math.sqrt(time_to_expiration / observation)) #SPOSTO LA BARRIERA PIU VICINA ALL'AUMENTARE DELLE OSSERVAZIONI COSI COSTA DI PIU (AL CONTRARIO: PUT)
                barrier_option_price = -spot_price * norm.cdf(-x1, 0, 1) * math.exp(-dividend_yield * time_to_expiration) + strike * math.exp(-interest_rate * time_to_expiration) \
                    * norm.cdf(-x1 + volatility * math.sqrt(time_to_expiration), 0, 1) + spot_price * math.exp(-dividend_yield * time_to_expiration) * ((barrier / spot_price)**(2 * lambdaa)) \
                        * ((norm.cdf(y, 0, 1) - norm.cdf(y1, 0, 1))) - strike * math.exp(-interest_rate * time_to_expiration) * ((barrier / spot_price)**((2 * lambdaa) -2)) * \
                             ((norm.cdf(y - volatility * math.sqrt(time_to_expiration), 0, 1) - norm.cdf(y1 - volatility * math.sqrt(time_to_expiration), 0, 1)))                          
            else:
                barrier2 = barrier * math.exp(-0.5826 * volatility * math.sqrt(time_to_expiration / observation)) #LONG BARRIER
                down_in_call_price = -spot_price * norm.cdf(-x1, 0, 1) * math.exp(-dividend_yield * time_to_expiration) + strike * math.exp(-interest_rate * time_to_expiration) \
                    * norm.cdf(-x1 + volatility * math.sqrt(time_to_expiration), 0, 1) + spot_price * math.exp(-dividend_yield * time_to_expiration) * ((barrier / spot_price)**(2 * lambdaa)) \
                        * ((norm.cdf(y, 0, 1) - norm.cdf(y1, 0, 1))) - strike * math.exp(-interest_rate * time_to_expiration) * ((barrier / spot_price)**((2 * lambdaa) -2)) * \
                             ((norm.cdf(y - volatility * math.sqrt(time_to_expiration), 0, 1) - norm.cdf(y1 - volatility * math.sqrt(time_to_expiration), 0, 1)))  
                barrier_option_price = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type)[0] - down_in_call_price


    return np.maximum(0,barrier_option_price)



def binomial_european_option(spot_price, strike, interest_rate, time_to_expiration, num_steps, up_factor, down_factor, dividend_yield, option_type ):
    
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
    d = 1 - down_factor  # Expected value in the down state I NEED TO DECIDE IF DIVIDE OR SUBTRACT
    dt = time_to_expiration / float(num_steps)
    df = np.exp(- (interest_rate - dividend_yield) * dt)
    qu = (np.exp((interest_rate - dividend_yield) * dt) - d) / (u - d)
    qd = 1 - qu
    
    # Calculate terminal stock prices-> THE RESULTS IS A VECTOR
    terminal_stock_prices = spot_price * (u ** np.arange(num_steps, -1, -1)) * (d ** np.arange(num_steps + 1))
    
    #calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(0, terminal_stock_prices - strike)
    else:
        payoffs = np.maximum(0, strike - terminal_stock_prices)
        
    #option price:
    for _ in range(num_steps):
        payoffs = (payoffs[:-1] * qu + payoffs[1:] * qd) * df

    return payoffs[0]


def binomial_american_option(spot_price, strike, interest_rate, time_to_expiration, num_steps, up_factor, down_factor, dividend_yield, option_type):
    
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
    terminal_stock_prices = [spot_price * (u ** np.arange(num_steps, -1, -1)) * (d ** np.arange(num_steps + 1))]

    # Simulate the possible stock prices path
    for _ in range(num_steps):
        prev_branches = terminal_stock_prices[-1]
        st = np.concatenate((prev_branches * u, [prev_branches[-1] * d]))
        terminal_stock_prices.append(st)

    # Compute the first payoff after appending new values to terminal_stock_prices
    if option_type == "call":
        payoffs = np.maximum(0, terminal_stock_prices[-1] - strike)
    else:
        payoffs = np.maximum(0, strike - terminal_stock_prices[-1])

    # Continue the computation of payoffs and terminal_stock_prices
    for i in range(num_steps):
        payoffs = (payoffs[:-1] * qu + payoffs[1:] * qd) * df

    # backwardation
    for i in reversed(range(num_steps)):
        # The payoffs from NOT exercising the option
        payoffs = (payoffs[:-1] * qu + payoffs[1:] * qd) * df
        # Payoffs from exercising, for American options
        if option_type == "call":
            payoffs = np.maximum(payoffs, terminal_stock_prices[i][: i + 1] - strike)
        else:
            payoffs = np.maximum(payoffs, strike - terminal_stock_prices[i][: i + 1])

    return payoffs[0]
