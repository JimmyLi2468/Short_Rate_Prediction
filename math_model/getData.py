import pandas as pd
from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
from numpy import array , hstack
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def p2f(x):
    return float(x.strip('%'))/100

#market = pd.read_csv('MarketData.csv', skiprows = [i for i in range(1,241)], converters={'10yr Bond':p2f})
market = pd.read_csv('MarketData.csv', skiprows = [i for i in range(1,337)])
eco = pd.read_csv('EconomicData.csv', skiprows = [i for i in range(1,337)])

short = eco['3m Rate (Secondary Market)'].copy(deep=True)
dates = eco['Sub Concept'].copy(deep=True)

eco.dropna(subset=['3m Rate (Secondary Market)'],inplace=True)
eco.drop(eco.columns[[0,2,3,4,5,6,8,9,10,14,17,18]],axis=1,inplace=True)
pmi = eco['PMI'].copy(deep=True)
cci = eco['UMICH Consumer Confidence'].copy(deep=True)
#eco.drop(columns = 'Potential NGDP', axis = 1, inplace=True)
#Core inflation is used by policymakers for the reason offered by Chairman Bernanke in the introduction—policymakers are most concerned about the future path of inflation, and current core inflation data may give better information than current headline data about future headline inflation. Headline inflation often does not have good predictive power over short-time periods because food and energy prices are so volatile. 

#print(eco.head())
#print(eco.columns)
#print(market.head())
normalized_eco = (eco - eco.mean())/(eco.max()-eco.min())
normalized_eco['PMI'] = pmi
normalized_eco['UMICH Consumer Confidence'] = cci
normalized_eco['10yr Bond'] = market['10yr Bond']
normalized_eco['Equity'] = market['US Equity']
normalized_short = (short - short.mean())/(eco.max()-eco.min())
#print(eco.head())


normalized_eco.fillna(method='ffill', inplace = True)
normalized_short.fillna(method='ffill', inplace=True)
#print(market)

liquidity = normalized_eco.iloc[:, [0,2,3,7,8]]
demand = normalized_eco.iloc[:, [1,4]]
index = normalized_eco.iloc[:, [5,6]]

print('economic data:')
print(eco.head())
print('\n normalized economic data: ')
print(normalized_eco.head())
print('\n indicators: ')
print(eco.columns)
print('\nshort rates:')
print(short)
print('\nliquidity data: ')
print( liquidity.head())
print('\n demand side data: ')
print(demand.head())
print('\n guide indexes: ')
print(index.head())

#plt.plot(normalized_eco['1yr-3m'])
#plt.xlabel('time step')
#plt.show()
#print(liquidity.head())
#print(liquidity['RGDP'][0:3])
