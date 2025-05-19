
import riskfolio as rp
import matplotlib.pyplot as plt
import pandas as pd

# pip install yfinance --upgrade --no-cache-dir
import yfinance as yf
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

yf.enable_debug_mode()

end=dt.datetime.now()
start="2024-01-01" 

if end.month<=6:
    start = end.replace(year=end.year-1, month=12+(end.month-6))
else:
    start = end.replace(month=end.month-6)

assets = ["BTC-USD","ETH-USD", "LTC-USD"]

data = yf.download(assets, start=start, end=end, auto_adjust=False)


returns = data['Adj Close'].pct_change().dropna()

# Setting Varibles for Model Caculations
method_mu = 'hist'
method_cov= 'hist'
hist= True
model =  'Classic'
rm= 'MV'
# 'obj' varible represent the object we are optimizing for. This can be things like Sharpe, Min Risk, Max Risk, Max Return, etc.
obj = 'Sharpe'
rf = 0
l = 0

# Optimization
port = rp.Portfolio(returns =returns)
port.assets_stats(method_mu=method_mu, method_cov=method_cov)
w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist )

# Plotting Optimization
ax = rp.plot_pie(w=w, title="Optimum Portfolio", others=0.05, cmap='tab20')
plt.show

# Plotting Efficient Frontier Mean - Standard Deviation (MV)
frontier = port.efficient_frontier(model=model, rm=rm, points=50, rf=rf, hist=hist)
ax = rp.plot_frontier(w_frontier=frontier, mu=port.mu, cov=port.cov, returns=returns, rm=rm, rf=rf, cmap='viridis', w=w)
plt.show()

# Just showing what the 'frontier' dataframe looks like
frontier


ax= rp.plot_frontier_area(w_frontier=frontier, cmap='tab20')
plt.show()

ax = rp.plot_pie(w=frontier[23], title="Optimum Portfolio", others=0.05, cmap='tab20')
plt.show

# Jupyter Report feature
# Note, need to change the 't_factor=365' and 'days_per_year=365' since crypto is traded 365 days a year.
ax = rp.jupyter_report(returns, w, rm=rm, t_factor=365, days_per_year=365)
plt.show()

ax = rp.jupyter_report(returns, frontier[23], rm=rm, t_factor=365, days_per_year=365)
plt.show()