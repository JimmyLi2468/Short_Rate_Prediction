import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import os
import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression
import math

#from cleanData import x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,y
from getData import eco, short, liquidity, demand, index, dates

# eco : Index([RGDP', 'Capacity Utiliization', '1yr-3m', '2yr-3m',
#       'Household Credit Creation', 'UMICH Consumer Confidence', 'PMI',
#       '10yr Bond', 'Equity'],
#      dtype='object')
# short : 3m Rate (Secondary Market)
# liquidity : Index(['RGDP', '1yr-3m', '2yr-3m', '10yr Bond', 'Equity'], dtype='object')
# demand : Index(['Capacity Utiliization', 'Household Credit Creation'], dtype='object')
# index : Index(['UMICH Consumer Confidence', 'PMI'], dtype='object')

def liquidity_proc(df, date):
    # total sum to 1
    pgdp= 0.05
    p1yr = 0.30
    p2yr = 0.25
    pequ = 0.20
    pbond = 0.2
    nextsix= [0]*6
    
    sumEquity = sum(df['Equity'][date-23:date-11])
    print('Equity Sum: ',sumEquity)
    # future rate effect
    # equity effect
    for i in range(6):
        #print(df['1yr-3m'][date-5+i],df['2yr-3m'][date-5+i])
        #nextsix[i] = (df['1yr-3m'][date-17+i] - df['1yr-3m'][date-5+i])*p1yr
        #nextsix[i] += (df['2yr-3m'][date-29+i]- df['2yr-3m'][date-5+i])*p2yr
        nextsix[i] = (df['1yr-3m'][date-5+i] - df['1yr-3m'][date-17+i])*p1yr
        nextsix[i] += (df['2yr-3m'][date-5+i]- df['2yr-3m'][date-29+i])*p2yr
        sumEquity
        nextsix[i] += (sumEquity+ df['Equity'][date-11+2*i] + df['Equity'][date-11+2*i+1]) * pequ     
    
    
    print('six month points after 3m + equity:\n\t ', nextsix)

    # gdp effect
    Q_1,Q_2,Q_3 = df['RGDP'][date], df['RGDP'][date-3], df['RGDP'][date-6]
    accelaration = (Q_3-Q_2) - (Q_2-Q_1)    
    
    print('accelaration: ', accelaration)

    

    for i in range(len(nextsix)):
        gdp_effect = (Q_2-Q_1) * (1+pgdp)*(1+accelaration)
        nextsix[i] += gdp_effect
    #if accelaration <= 0:
    #    for i in range(len(nextsix)):
    #        if nextsix[i] <=0:
    #            gdp_effect = (Q_2-Q_1)*(1-accelaration)*(1+pgdp)
    #        else:
    #            gdp_effect = (Q_2-Q_1)*(1+pgdp)
    #        nextsix[i] += gdp_effect
    #else:
    #    for i in range(len(nextsix)):
    #        if nextsix[i] >0:
    #            gdp_effect = (Q_2-Q_1) * (1+pgdp)*(1+accelaration)
    #        else:
    #            gdp_effect = (Q_2-Q_1)*(1+pgdp)    
    #        nextsix[i] += gdp_effect

    print('\nsix month points with gdp effect:\n\t ', nextsix)
    # liquidity decreasing
    # 10yr bond decreasing effect on 3m may persist for a while
    # assuming for 2 months
    print('10yr bond this month: ', df['10yr Bond'][date])
    if df['10yr Bond'][date] <= 0:
        nextsix[1] += df['10yr Bond'][date]/5
        nextsix[2] += df['10yr Bond'][date]
    else:
        nextsix[0] += df['10yr Bond'][date]
        nextsix[1] += df['10yr Bond'][date]/5


    print('\nsix month with bond: \n\t', nextsix)
    return nextsix

# process demand side moving direction
def demand_proc(df, date):
#    print(df['Capacity Utiliization'][date-5:date+1])
#    print(df['Capacity Utiliization'][date-11:date-5])
    # utilization part
    util_now = sum(df['Capacity Utiliization'][date-5:date+1])
    util_before = sum(df['Capacity Utiliization'][date-11:date-5])

    util_coef = (util_now - util_before)/6

    # household credit part
    credit_now = sum(df['Household Credit Creation'][date-5:date+1])
    credit_before = sum(df['Household Credit Creation'][date-11:date-5])
    #print(half_now, half_before)
    credit_coef = (credit_now - credit_before)/6
    
    total_coef = util_coef*0.5 + credit_coef*0.5
    print('demand side coefficient: ', total_coef)

    return total_coef
    
    # if decreasing utilization
    #if half_now - half_before <= 0:
             

def validate(liq, dem, index, short,date ):
    
    pmi_avg = sum(index['PMI'][date-5:date+1])/6
    cci_dir = index['UMICH Consumer Confidence'][date-5:date+1].pct_change().sum()
    #print(index['UMICH Consumer Confidence'][date-5:date+1].pct_change())
    print('pmi half year average: ',pmi_avg)
    print('cci half year direction: ', cci_dir)
    #print(short[date]-short[date-1])
    if short[date] - short[date-1] == 0.0:
        multiplier = 0.0001
    else:
        order = math.floor(math.log10(abs(short[date]-short[date-1])))
        multiplier = math.pow(10, order)
    guide1 = [(pmi_avg-50)*multiplier+short[date]]*7
    guide2 = [short[date]]
    for i in range(1,7):
        guide2.append(guide2[i-1]+(cci_dir*multiplier)*i)

    n = 36
    simpleMA = short[date-n:date+1].rolling(window=6).mean()
    simpleMA = simpleMA[~np.isnan(simpleMA)]
    #print(simpleMA)
    dates = []
    for i in range(len(simpleMA)):
        dates.append(date - len(simpleMA)+i+1)
    x = np.array(dates).reshape((-1,1))
    model = LinearRegression().fit(x, simpleMA)
    #print(model.coef_)


    linearmodel = []
    output = []
    output.append(liq[0])
    for i in range(1,len(liq)):
        regression = model.intercept_ + model.coef_*(date+i)
        temp1 = abs(liq[i]+dem-regression)
        temp2 = abs(liq[i]-regression)
        if temp1>temp2:
            #print('liquidity more')
            output.append(output[i-1]*(1+liq[i]*0.7+dem*0.3))
        else:
            #print('demand more')
            output.append(output[i-1]*(1+liq[i]*0.3+dem*0.7))
    print('\nfinal prediction: \n\t', output)
    return guide1, guide2, output
    #plt.plot(dates,short[25:51],label='short')
    #plt.plot(dates,simpleMA, label='MA')
    #plt.plot(dates, linearmodel,label='regression')
    #plt.legend()
    #plt.show()
    
   
    

#start = 50
#print(dates[start])
#x = [i for i in range(start, start+7)]
#liq = liquidity_proc(liquidity, start)
#liq.insert(0,short[start])

#for i in range(1,len(liq)):
#    liq[i] = liq[i-1]*(1+liq[i])

#dem = demand_proc(demand,start)
#guide1, guide2, output = validate(liq,dem,index, short, start)

#print('\nactual six months: \t\n', short[start:start+7])    

#plt.plot(x,short[start:start+7], label='actual')
#plt.plot(x,liq, label='prediction')
#plt.plot(x, guide1, label='pmi guide')
#plt.plot(x, guide2, label='cci guide')
#plt.plot(x, output, label='combined output')
#plt.legend(loc = 'upper right')
#plt.plot(x, liquidity['10yr Bond'][start:start+7])
#plt.show()


#fig, ax = plt.subplots()

#ax.plot(dates, short, label='3m short rate', color='blue')

dates = dates.append(pd.Series([1,2,3,4,5,6], index=[501,502,503,504,505,506]))
outputs = [None]*len(dates)
guides1 = [None]*len(dates)
guides2 = [None]*len(dates)
rmse = []
#print(dates)
start = 36
while start+5 < len(short):
    print('====================')
    print('current short rate: ', short[start])
    liq = liquidity_proc(liquidity, start)
    liq.insert(0,short[start])
    dem = demand_proc(demand,start)
    g1, g2, out = validate(liq,dem,index, short, start)
    temp = 0
    for i in range(len(out)):
        outputs[start+i] = out[i]
        guides1[start+i] = g1[i]
        guides2[start+i] = g2[i]
        temp += (out[i]-short[start+i])**2
    rmse.append(sqrt(temp/len(out))*100)
    start += 12

start=500
print('=====================')
print('current short rate: ', short[start])
liq = liquidity_proc(liquidity, start)
liq.insert(0,short[start])
dem = demand_proc(demand,start)
g1, g2, out = validate(liq,dem,index, short, start)
for i in range(len(out)):
    outputs[start+i] = out[i]
    guides1[start+i] = g1[i]
    guides2[start+i] = g2[i]

print('========RMSE========')
print(rmse)
print(sum(rmse)/len(rmse))

#print(len(outputs))
#ax.plot(dates, outputs, label='prediction')
#ax.plot(dates, guides1, label='PMI guide')
#ax.plot(dates, guides2, label='CCI guide')
df = pd.DataFrame({"dates":dates, 
                    "3m short rate": short, 
                    "prediction": outputs,
                    "PMI Guide": guides1,
                    "CCI Guide": guides2
    })
df.plot(x='dates')
#plt.xlabel('dates')
#plt.ylabel('rates')
plt.legend(loc='upper right')
plt.title('Short rate prediction model ')
plt.show()


