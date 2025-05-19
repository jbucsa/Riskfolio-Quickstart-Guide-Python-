
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

# Download data with proper settings
# try:
#     data = yf.download(tickers=assets, start=start, end=end, group_by='ticker', auto_adjust=True)
#     # Extract the 'Close' prices for each ticker
#     prices = data.xs('Close', axis=1, level=1)
# except Exception as e:
#     print("Download failed:", e)
#     prices = pd.DataFrame()

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