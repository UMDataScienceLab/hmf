import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch

import matplotlib.dates as mdates

def readdata():
    df=pd.read_csv('data/stock/daily_data_1990.csv',sep=',')
    
    return df

def df2data(df):
    delta = 30
    start = 0
    Y = []   
    
    fulldata = df.iloc[:,2:].to_numpy()
    N = len(fulldata[0])//delta 

    for i in range(N):
        slicei = fulldata[:,start:(start+delta)]  

        Y.append(slicei)
        
        start += delta
    return Y
 

def loadstocks(device):
    df = readdata()
    Y = df2data(df)    
    Y=[torch.tensor(dfi,device=device).float() for dfi in Y]
    return Y, df



def plotall(rawdf, lf):
    fig, ax = plt.subplots()

    
    score = torch.cat(lf).detach().numpy()
    years = pd.to_datetime(list(rawdf.columns.values)[2:], format='%Y-%m-%d')
    commence = 2000
    fini = 6000#len(score)
    ax.plot(years[commence:fini], score[commence:fini], label="Anomaly Score")
    ax.tick_params(axis='y')

    ax2 = ax.twinx()
    dfsp500=pd.read_csv('stock/sp500.csv',sep=',')
    sp500 = dfsp500['Closing'].to_numpy()
    ax2.plot(years[commence:fini], sp500[commence:fini], color='orange', label='SP500')
    ax2.tick_params(axis='y')
    plt.legend()
    plt.xticks(rotation=-45)
    plt.gcf().autofmt_xdate()

    plt.savefig('stock.png',bbox_inches='tight')
    return
    
        
def see_stocks():
    df=pd.read_csv('stock/sp500.csv',sep=',')
    print(df['Return'])
    plt.plot(df['Closing'])
    plt.savefig('stock.png',bbox_inches='tight')

