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


def digital_pricer(spot_price, strike, coupon, time_to_expiration, volatility, interest_rate, dividend_yield, barrier_shift, width_adjustment, option_type, options_style):
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
        spot1 = spot_price - coupon / 2 + barrier_shift + width_adjustment
        price1 = calculate_options_price(spot1, strike, time_to_expiration, volatility, interest_rate, dividend_yield, "call")[0]
        spot2 = spot_price + coupon / 2 + barrier_shift - width_adjustment
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

    return digital_price, theo_price

       
d1 = (math.log(100 / 110) + (0.03  + (0.2 ** 2) / 2) * 0.5) / (0.2 * math.sqrt(0.5))
d2 = d1 - 0.2 * math.sqrt(0.5) 
X=digital_pricer(100, 110, 10, 0.5, 0.2, 0.03, 0, 0,0, "call", "american")
Y= 10 * norm.cdf(d2, 0, 1)*math.exp(-0.03 * 0.5)*2
print(Y)
print(X)

#THIS CODE IS VALIDE ONLY FOR REVERSE BARRIER UP CALL AND DOWN PUT

def barrier_continous_pricer(spot_price, strike, barrier, rebate, time_to_expiration, volatility, interest_rate, dividend_yield, option_type, barrier_type, observation ):
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

pra = barrier_continous_pricer(100, 100, 80, 0, 1, 0.3, 0.02, 0, "put", "0ut", 1 )
    
print(pra)