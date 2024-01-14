import blackscholes as bl

spot_price=100
strike=105
time_to_expiration=1
volatility=0.3
interest_rate=0.02
dividend_yield=0.01


option = bl.BlackScholesCall (spot_price, strike, time_to_expiration, volatility, interest_rate, dividend_yield)
Price = option.price()
Delta = option.delta()
Gamma = option.gamma()
Vega = option.vega()
Theta = option.theta()
Rho = option.rho()

print(Price, Delta, Gamma, Vega, Theta, Rho)