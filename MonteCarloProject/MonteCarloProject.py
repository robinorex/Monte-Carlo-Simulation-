import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import datetime as dt
# from scipy.stats import norm, t
from pandas_datareader import data as pdr
import yfinance as yf
def get_data(stocks,start,end):
    yf.pdr_override()
    stock_Data=pdr.get_data_yahoo(stocks,start,end)
    stock_Data=stock_Data["Close"]
    returns= stock_Data.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix
stock_list= ["CBA","BHP","TLS","NAB","WBC","STO","CSL","ANZ","FMG","WES"]
stocks= [stock+".AX" for stock in stock_list]
endDate= dt.datetime.now()
startDate= endDate - dt.timedelta(days=300)
meanReturns,covMatrix=get_data(stocks,startDate,endDate)
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights) #10x1
# print(weights)
#Monte Carlo Method
mc_sims = 10000  #number of simulations
T = 360  #timeframe in days
meanM = np.full(shape=(T,len(weights)),fill_value=meanReturns)
meanM = meanM.T
portfolio_sims= np.full(shape=(T,mc_sims),fill_value=0.0)
initialPortfolio = 10000
for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))  #Z is samples from normal distribution
    L = np.linalg.cholesky(covMatrix)  #Cholesky Decomposition, L=Lower Triangular Matrix
    dailyReturns = meanM + np.inner(L, Z)   # L: 10x10  Z: 10x100  meanM: 10x100 dailyReturs: 10x100
    portfolio_sims[:,m] = np.cumprod(np.inner(weights,dailyReturns.T)+1)*initialPortfolio  #10x100
plt.plot(portfolio_sims)
plt.ylabel("Portfolio Value ($)")
plt.xlabel("Days")
plt.title("Monte Carlo Simulation of Stock Portfolio")
plt.show()