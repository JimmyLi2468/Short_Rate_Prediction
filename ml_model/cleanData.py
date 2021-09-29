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
market = pd.read_csv('MarketData.csv', skiprows = [i for i in range(1,156)])

#market[market.columns['10yr Bond']] = df[df.columns[1:]].replace('[\$,]%', '', regex=True).astype(float)

eco = pd.read_csv('EconomicData.csv', skiprows = [i for i in range(1,156)])


short = eco['3m Rate (Secondary Market)'].copy(deep=True)

market.drop(market.columns[[1,3,5,6]],axis=1,inplace=True)

eco.drop(eco.columns[[7,10,15,17]],axis=1,inplace=True)
eco.drop(columns = 'Potential NGDP', axis = 1, inplace=True)
#Core inflation is used by policymakers for the reason offered by Chairman Bernanke in the introduction—policymakers are most concerned about the future path of inflation, and current core inflation data may give better information than current headline data about future headline inflation. Headline inflation often does not have good predictive power over short-time periods because food and energy prices are so volatile. 
eco.drop(columns = 'Headline CPI', axis = 1, inplace=True)
eco.drop(columns = 'CMD Price Index', axis = 1, inplace=True)
eco.drop(columns = 'Business Credit Creation', axis = 1, inplace=True)
#eco.drop(columns = 'Real PCE Growth', axis = 1, inplace=True)
eco.drop(columns = '2yr-3m', axis = 1, inplace=True)

eco['10yr Bond'] = market['10yr Bond']

eco.fillna(method='ffill', inplace = True)
short.fillna(method='ffill', inplace=True)
#print(market)

print(eco.columns)
print(market.columns)
#print(eco)
#print(short)

# Step 1: Separate Data and reshape to [rows, columns] structure
# shape => [datapoints, 1]
x1 = eco[eco.columns[1]].values
x1 = x1.reshape((len(x1), 1))

x2 = eco[eco.columns[2]].values
x2 = x2.reshape((len(x2), 1))

x3 = eco[eco.columns[3]].values
x3 = x3.reshape((len(x3), 1))

x4 = eco[eco.columns[4]].values
x4 = x4.reshape((len(x4), 1))

x5 = eco[eco.columns[5]].values
x5 = x5.reshape((len(x5), 1))

x6 = eco[eco.columns[6]].values
x6 = x6.reshape((len(x6), 1))

x7 = eco[eco.columns[7]].values
x7 = x7.reshape((len(x7), 1))

x8 = eco[eco.columns[8]].values
x8 = x8.reshape((len(x8), 1))

x9 = eco[eco.columns[9]].values
x9 = x9.reshape((len(x9), 1))

x10= eco[eco.columns[10]].values
x10= x10.reshape((len(x10), 1))

#print(x1.head(), x2.head(), x3.head(), x4.head(), x5.head(), x6.head(), x7.head(), x8.head(), x9.head(), x10.head())
#print(x10[:15])
print(x1.shape)

y = short.values
y = y.reshape((len(y), 1))
print(y.shape)

#plt.plot(y)
#plt.xlabel('time step')
#plt.show()
