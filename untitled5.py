import math
from scipy.stats import norm
import numpy as np

def calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type):


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



def barrier_continous_pricer(spot_price, strike, barrier, rebate, time_to_expiration, volatility, interest_rate, dividend_yield, option_type, barrier_type, observation):
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

    # Constants
    lambdaa = (interest_rate - dividend_yield + ((volatility**2) / 2)) / volatility**2
    sqrt_time = volatility * math.sqrt(time_to_expiration)
    y = (math.log((barrier**2) / (strike * spot_price)) / sqrt_time) + lambdaa * sqrt_time
    x1 = (math.log(strike / barrier) / sqrt_time) + lambdaa * sqrt_time
    y1 = (math.log(barrier / spot_price) / sqrt_time) + lambdaa * sqrt_time

    # Function for option calculation
    def calculate_option_price(x, y, x1, y1, barrier_ratio, barrier_option_multiplier):
        return barrier_option_multiplier * (
            x * norm.cdf(x1, 0, 1) * math.exp(-dividend_yield * time_to_expiration)
            - strike * math.exp(-interest_rate * time_to_expiration) * norm.cdf(x1 - sqrt_time, 0, 1)
            - spot_price * math.exp(-dividend_yield * time_to_expiration) * (barrier_ratio**(2 * lambdaa))
            * (norm.cdf(-y, 0, 1) - norm.cdf(-y1, 0, 1))
            + strike * math.exp(-interest_rate * time_to_expiration) * (barrier_ratio**((2 * lambdaa) - 2))
            * (norm.cdf(-y + sqrt_time, 0, 1) - norm.cdf(-y1 + sqrt_time, 0, 1))
        )

    # Main logic
    if observation == 0:
        if option_type == "call":
            barrier_option_price = calculate_option_price(x1, y, x1 - sqrt_time, y1, barrier / spot_price, 1)
            if barrier_type == "out":
                barrier_option_price = calculate_option_price(x1, y, x1 - sqrt_time, y1, barrier / spot_price, 1)
                barrier_option_price = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type)[0] - barrier_option_price
        else:
            barrier_option_price = calculate_option_price(-x1, -y, -x1 + sqrt_time, -y1, barrier / spot_price, -1)
            if barrier_type == "out":
                barrier_option_price = calculate_option_price(-x1, -y, -x1 + sqrt_time, -y1, barrier / spot_price, -1)
                barrier_option_price = calculate_options_price(spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield, option_type)[0] - barrier_option_price
    else:
        if option_type == "call":
            barrier_ratio = barrier * math.exp(-0.5826 * volatility * math.sqrt(time_to_expiration / observation)) if barrier_type == "in" else barrier * math.exp(0.5826 * volatility * math.sqrt(time_to_expiration / observation))
        else:
            barrier_ratio = barrier * math.exp(0.5826 * volatility * math.sqrt(time_to_expiration / observation)) if barrier_type == "in" else barrier * math.exp(-0.5826 * volatility * math.sqrt(time_to_expiration / observation))

        barrier_option_price = calculate_option_price(x1, y, x1 - sqrt_time, y1, barrier_ratio, 1)

    return barrier_option_price




def barrier_continous_pricer_new(spot_price, strike, barrier, rebate, time_to_expiration, volatility, interest_rate, dividend_yield, option_type, barrier_type, observation ):
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


    return barrier_option_price

pra = barrier_continous_pricer(100, 100, 80, 0, 1, 0.3, 0.02, 0, "put", "out", 1 )
pra2 = barrier_continous_pricer_new(100, 100, 80, 0, 1, 0.3, 0.02, 0, "put", "out", 1 )
print(pra,pra2)


