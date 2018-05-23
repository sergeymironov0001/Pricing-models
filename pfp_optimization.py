import numpy as np
import math
from scipy.optimize import minimize

BASE_TOL = 1e-9
MAX_LOG_RATE = 1e3

def find_best_wghts(total):

    n_prods = total.shape[2]
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(np.array(x))-1})
    bnds = (np.repeat([np.array([0,1])],n_prods,axis =0))

    #------init sets-------------------------------------------
    
    ww = []
    for i in range(0,n_prods):
        weights = np.zeros(n_prods)
        weights[i] = 1
        ww.append(weights)

    ww1 = np.ones(n_prods)/n_prods
    ww.append(ww1)

    for i in range(0,n_prods):
        ww2 = ww1/2
        ww2[i] = ww2[i] + 0.5
        ww.append(ww2)
        
    #------init sets-------------------------------------------
       
    func = 10**6
    best_weights = np.ones(n_prods)/n_prods
    
    for weights in ww:
        res = minimize(utility, weights, args = (total,0), method='SLSQP', bounds=bnds, constraints=cons, options={'disp': False}, tol = 1)
        if res.fun < func:
            func = res.fun
            best_weights = res.x
    
    res2 = minimize(utility, best_weights, args = (total,0), method='SLSQP', bounds=bnds, constraints=cons, options={'disp': False}, tol = 0.01)            

    return res2.x, res2.fun 
#    return best_weights, func

def utility(weights, total, option=0):       

    r = np.apply_along_axis(irr_newton, 0, np.sum( total*weights ,axis = 2) )     

    income = np.mean(r)    
    risk = 0.01
    if len(r[r<0]) > 0: risk = np.abs(np.mean(r[r<0])) + 0.01
    
    #risk = max(0.01,-np.percentile(r,1))
    #if len(set(r)) > 3:
     #   risk = np.std(r)

    if option == 1: return income
    if option == 2: return risk
    #else: return -np.sign(income)*(np.abs(income))**0.5/risk
    else: return - income / risk

def find_r(total, weights, disp = False):    
    return np.apply_along_axis(irr_newton, 0, np.sum(total*weights,axis = 2)) 

def irr_newton(stream, tol=BASE_TOL):
    rate = 0.0
    r = np.arange(len(stream))
    
    for steps in range(50):        
        # Factor exp(m) out of the numerator & denominator for numerical stability
        m = max(-rate * r)
        f = np.exp(-rate * r - m)
        t = np.dot(f, stream)
        if abs(t) < tol * math.exp(-m):
            break
        u = np.dot(f * r, stream)
        # Clip the update to avoid jumping into some numerically unstable place
        rate = rate + np.clip(t / u, -1.0, 1.0)

    return (math.exp(rate))**12 - 1

def irr_binary_search(stream, tol=BASE_TOL):
    rate_lo, rate_hi = -MAX_LOG_RATE, +MAX_LOG_RATE
    sgn = np.sign(stream[0]) # f(x) is decreasing
    for steps in range(100):
        rate = (rate_lo + rate_hi)/2
        r = np.arange(len(stream))
        # Factor exp(m) out because it doesn't affect the sign
        m = max(-rate * r)
        f = np.exp(-rate * r - m)
        t = np.dot(f, stream)
        if abs(t) < tol * math.exp(-m):
            break
        if t * sgn > 0:
            rate_hi = rate
        else:
            rate_lo = rate
    rate = (rate_lo + rate_hi) / 2
    return math.exp(rate) - 1