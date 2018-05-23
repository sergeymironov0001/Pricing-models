import numpy as np
import pandas as pd
from datetime import datetime
import random
from sklearn.linear_model import LinearRegression



def set_prices(n_points, n_sims, mu, sigma, corr_matrix, dt):
    
    #np.random.seed(np.random) 
    rndm = np.random.standard_normal(size = (n_points * n_sims, n_assets))    
    corr_values = rndm.dot(cholesky(corr_matrix, lower=False))*sigma 
    returns = np.cumsum(((mu-0.5*sigma**2)*dt + np.sqrt(dt)*corr_values).reshape(n_points, n_sims, n_assets), axis = 0)
    prices = np.exp(returns)

    return prices

def ba_scenarios(BAs, simulation_years, points_in_year, n_scenarios, assumptions, assumptions_flag):

    n_timepoints = points_in_year*simulation_years
    dT = 1/points_in_year

    #Hist = pd.read_csv('quotes_Funds&RFactors4.csv',  sep = ';',decimal = ',')
    Hist = pd.read_excel('quotes_Funds&RFactors7.xlsx', decimal = '.')
    Hist = Hist.interpolate('linear', axis  = 0) 
    Hist.index = Hist['Date']
    Hist = Hist.drop(['Date'], axis = 1)

    #Hist.index = [datetime.strptime(date, '%d.%m.%Y').date() for date in list(Hist.index)]

    Returns = np.log(Hist/Hist.shift(1))
    Returns = Returns.dropna()
    
    predictors = Returns.columns[:20]   
    RF_yields = Returns[predictors].mean()*52
    
    if (assumptions_flag==1):
        RF_yields[assumptions.index] = assumptions['return']
    
    Means = Returns[BAs].mean()*52
    linreg = LinearRegression(normalize=False, fit_intercept=True)
    
    for ba in BAs:        
        linreg.fit(Returns[predictors],Returns[ba])
        Means[ba] = sum(linreg.coef_*RF_yields)+linreg.intercept_*52    
    Sigmas = Returns[BAs].cov()*52

    Gens = np.random.multivariate_normal(Means*dT, Sigmas*dT, n_scenarios*n_timepoints).reshape(n_timepoints, n_scenarios, len(BAs))

    x = np.exp(np.cumsum(Gens, axis = 0))

    return x