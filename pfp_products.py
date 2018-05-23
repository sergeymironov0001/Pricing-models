import numpy as np
import pandas as pd
import plotly 
import plotly.plotly as py
import matplotlib
import matplotlib.mlab as mlab
import seaborn as sns
import random
import time
import datetime
import matplotlib.pyplot as plt

from timeit import default_timer as timer

from datetime import datetime
from scipy import stats
from plotly import figure_factory as FF
from scipy.linalg import svd, sqrtm, cholesky
from scipy.stats import multivariate_normal, normaltest, shapiro


from pandas.tools.plotting import scatter_matrix
from pandas import bdate_range
from pandas_datareader.data import DataReader
from pandas import Panel, DataFrame

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

Days_in_year = 252


def read_prodcat(file_name):
    xls = pd.ExcelFile(file_name)

    sheet_to_df_map = {}
    und_list = []
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        df = df.dropna(how='all')
        df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
        sheet_to_df_map[sheet_name] = df

        und_cols = [col for col in df.columns if 'und' in col]
        if 'und_curr' in und_cols: und_cols.remove('und_curr')
        for col in und_cols:
            und_list.extend(df[col])
    und_list = [x for x in list(set(und_list)) if str(x) != 'nan']
    return (sheet_to_df_map, xls.sheet_names, und_list)

def read_buyouts(file_name, points_in_year, Years):
    xls = pd.ExcelFile(file_name)

    df = xls.parse()
    df = df.dropna(how='all')
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
    buyouts = df
    
    prev_term = 0
    
    # срок в каждой строке должен быть строго больше предыдущего. проверка нужна, но сейчас без нее
    
    for index, row in buyouts.iterrows():
        number_of_lines = (buyouts['term, years'][index] - prev_term)*points_in_year

        a = np.array(row[1:])
        
        b = np.repeat([a],number_of_lines,axis = 0)
        if prev_term == 0:
            bs = b
        if prev_term > 0:
            bs = np.concatenate((bs,b), axis = 0)

        prev_term = buyouts['term, years'][index]

    # сроки выше существующих в таблице дополним до строка симуляции

    c = np.repeat([bs[-1,:]],(Years - prev_term)*points_in_year, axis = 0)
    bs = np.concatenate((bs,c), axis = 0)    
    return (bs, np.delete(buyouts.columns.values,0))


def margin(prod_cat, prod_types):
    margin = []
    for i in prod_types:        
        prods = prod_cat[i]
        margin.extend((prods.margin))        
    return(np.array(margin))      

def calc_prod_deposits(prods, N_sims, Years, points_in_year, client_term, prices, symbols):
    # считаем депозиты

    n_periods = Years * points_in_year

    cf_rub = np.zeros((n_periods, prods.shape[0]))
    cf_usd = np.zeros((n_periods, prods.shape[0]))

    ### для каждого продукта отдельно
    for index, row in prods.iterrows():

        ## купонные даты
        cdates = np.arange(int(points_in_year/row.freq - 1), 
                       int(points_in_year*row.term),
                       int(points_in_year/row.freq))
        ## дата номинала
        ndate = row.term*points_in_year - 1

        cf_rub[cdates,index] = cf_rub[cdates,index] + row.rate / row.freq * (row.currency == 'RUB')
        cf_usd[cdates,index] = cf_usd[cdates,index] + row.rate / row.freq * (row.currency == 'USD')

        cf_rub[ndate,index] = cf_rub[ndate,index] + 1 * (row.currency == 'RUB')
        cf_usd[ndate,index] = cf_usd[ndate,index] + 1 * (row.currency == 'USD')

    prods_rub = np.repeat(cf_rub[:, np.newaxis, :], N_sims, axis=1)
    prods_usd = np.repeat(cf_usd[:, np.newaxis, :], N_sims, axis=1)

    return prods_rub, prods_usd, list(prods.name.values)
    
def calc_prod_pifpaf(prods, N_sims, Years, points_in_year, client_term, prices, symbols):
    # это ложка для ПИФа
    n_periods = Years * points_in_year
    
    # список базовых активов
    positions = []
    for a in prods.und:
        positions.append(np.where(symbols == a)[0][0])

    ### разбиваем на рубли и доллары,

    pr = np.multiply(prices[:,:,positions], np.array(prods.leverage))

    prods_usd = np.multiply(pr, np.array(prods.currency == 'USD'))  
    prods_rub = np.multiply(pr, np.array(prods.currency == 'RUB')) 

    term_table = np.zeros((n_periods, prods.shape[0]))
    term_table [(client_term * points_in_year - 1), :] = 1
    term_table = np.repeat(term_table[:, np.newaxis, :], N_sims, axis=1)

    ### и оставляем только на нужный срок
    prods_rub = np.multiply(prods_rub, term_table)
    prods_usd = np.multiply(prods_usd, term_table)    

    return prods_rub, prods_usd, list(prods.name.values)

def calc_prod_exchange_structured_bonds(prods, N_sims, Years, points_in_year, client_term, prices, symbols):

    #БСО
    #prods = prod_cat['исж классический']

    # формируем список базовых активов
    n_periods = Years * points_in_year

    positions = []
    for a in prods.und:
        positions.append(np.where(symbols == a)[0][0])

    # разбиваем на рубли и доллары, гарантийный и рисковый фонд

    ## рисковый фонд
    ### применяем КУ (плечо), отрезаем движение вниз
    p1 = np.maximum(np.multiply((prices[:,:,positions]-1), np.array(prods.leverage)),0)

    ### разбиваем РФ на рубли и доллары,
    prods_usd_risk = np.multiply(p1, np.array(prods.und_curr == 'USD'))  
    prods_rub_risk = np.multiply(p1, np.array(prods.und_curr == 'RUB')) 

    ### достаем выкупные из отдельного файла, приводим массив к виду n_periods x n_prods
    bs, buyouts = read_buyouts('buyoutscheme.xlsx', points_in_year, Years)
    positions_bs = []
    for a in prods.buyout:
        positions_bs.append(np.where(buyouts == a)[0][0])
    bs = bs[:,positions_bs]  

    ## гарантия
    guar = np.multiply(bs, np.array(prods.guarantee))
    prods_usd_guar = np.multiply(guar, np.array(prods.currency == 'USD'))  
    prods_rub_guar = np.multiply(guar, np.array(prods.currency == 'RUB'))

    ##срок - либо срок продукта (со штрафом), либо срок, нужный клиенту (меньший из двух)
    term_row = (np.minimum(np.array(prods.term), client_term) * points_in_year - 1)
    term_table = np.zeros((n_periods, prods.shape[0]))

    for i in range(prods.shape[0]):
        term_table [int(term_row[i]), i] = 1
    term_table = np.repeat(term_table[:, np.newaxis, :], N_sims, axis=1)

    ## собираем все вместе:
    ### сначала рисковый и гарантийный фонд
    prods_rub_1 = prods_rub_risk + np.repeat(prods_rub_guar[:, np.newaxis, :], N_sims, axis=1)
    prods_usd_1 = prods_usd_risk + np.repeat(prods_usd_guar[:, np.newaxis, :], N_sims, axis=1)
    
    ### и оставляем только на нужный срок
    prods_rub = np.multiply(prods_rub_1, term_table)
    prods_usd = np.multiply(prods_usd_1, term_table)    

    return prods_rub, prods_usd, list(prods.name.values)


def calc_prod_ili_classic(prods, N_sims, Years, points_in_year, client_term, prices, symbols, response_option = 0, response_option_value = 0):

    #1 - исж классический
    #prods = prod_cat['исж классический']

    # формируем список базовых активов
    n_periods = Years * points_in_year

    positions = []
    for a in prods.und:
        positions.append(np.where(symbols == a)[0][0])

    # разбиваем на рубли и доллары, гарантийный и рисковый фонд

    ## рисковый фонд
    ### применяем КУ (плечо), отрезаем движение вниз
    if response_option == 2: 
        prev_lev = prods.leverage[0]
        prods.leverage[0] = response_option_value * prods.leverage[0]
        print('leverage adjusted by', response_option_value, 'from', prev_lev, 'to ', prods.leverage[0], end="")

    p1 = np.maximum(np.multiply((prices[:,:,positions]-1), np.array(prods.leverage)),0)

    ### разбиваем РФ на рубли и доллары,
    prods_usd_risk = np.multiply(p1, np.array(prods.und_curr == 'USD'))  
    prods_rub_risk = np.multiply(p1, np.array(prods.und_curr == 'RUB')) 

    ### достаем выкупные из отдельного файла, приводим массив к виду n_periods x n_prods
    bs, buyouts = read_buyouts('buyoutscheme.xlsx', points_in_year, Years)
    positions_bs = []
    for a in prods.buyout:
        positions_bs.append(np.where(buyouts == a)[0][0])
    bs = bs[:,positions_bs]  

    ## гарантия
    guar = np.multiply(bs, np.array(prods.guarantee))
    prods_usd_guar = np.multiply(guar, np.array(prods.currency == 'USD'))  
    prods_rub_guar = np.multiply(guar, np.array(prods.currency == 'RUB'))

    ##срок - либо срок продукта (со штрафом), либо срок, нужный клиенту (меньший из двух)
    term_row = (np.minimum(np.array(prods.term), client_term) * points_in_year - 1)
    term_table = np.zeros((n_periods, prods.shape[0]))

    for i in range(prods.shape[0]):
        term_table [int(term_row[i]), i] = 1
    term_table = np.repeat(term_table[:, np.newaxis, :], N_sims, axis=1)

    ## собираем все вместе:
    ### сначала рисковый и гарантийный фонд
    prods_rub_1 = prods_rub_risk + np.repeat(prods_rub_guar[:, np.newaxis, :], N_sims, axis=1)
    prods_usd_1 = prods_usd_risk + np.repeat(prods_usd_guar[:, np.newaxis, :], N_sims, axis=1)
    
    ### и оставляем только на нужный срок
    prods_rub_1 = np.multiply(prods_rub_1, term_table)
    prods_usd_1 = np.multiply(prods_usd_1, term_table)    

    return prods_rub_1, prods_usd_1, list(prods.name.values)

def calc_prod_ili_coupon(prods, N_sims, Years, points_in_year, client_term, prices, symbols, response_option = 0, response_option_value = 0):
    
    #2 - исж купонный
    ### достаем выкупные из отдельного файла, приводим массив к виду n_periods x n_prods
    n_periods = Years * points_in_year

    bs, buyouts = read_buyouts('buyoutscheme.xlsx', points_in_year, Years)
    positions_bs = []
    for a in prods.buyout:
        positions_bs.append(np.where(buyouts == a)[0][0])
    bs = bs[:,positions_bs]  

    ## гарантия
    guar = np.multiply(bs, np.array(prods.guarantee))
    prods_usd_guar = np.multiply(guar, np.array(prods.currency == 'USD'))  
    prods_rub_guar = np.multiply(guar, np.array(prods.currency == 'RUB'))

    ##срок - либо срок продукта (со штрафом), либо срок, нужный клиенту (меньший из двух)
    term_row = (np.minimum(np.array(prods.term), client_term) * points_in_year - 1)
    term_table = np.zeros((n_periods, prods.shape[0]))

    for i in range(prods.shape[0]):
        term_table [int(term_row[i]), i] = 1
    term_table = np.repeat(term_table[:, np.newaxis, :], N_sims, axis=1)

    prods_rub_2 = np.multiply(np.repeat(prods_rub_guar[:, np.newaxis, :], N_sims, axis=1), term_table)
    prods_usd_2 = np.multiply(np.repeat(prods_usd_guar[:, np.newaxis, :], N_sims, axis=1), term_table)

    ## купон
    cf_rub = np.zeros(prods_rub_2.shape)
    cf_usd = np.zeros(prods_usd_2.shape)

    ### для каждого продукта отдельно
    for index, row in prods.iterrows():    
        term = min(row.term,client_term)
        cdates = range(int(points_in_year/row.freq)-1,points_in_year*term,int(points_in_year/row.freq) ) 
        und = pd.DataFrame(row[list(row.drop('und_curr').filter(regex='und'))].index).dropna()[0].values

        positions = []
        for a in und:
            positions.append(np.where(symbols == a)[0][0])

        # пробит страйк по худшему активу
        coupon_flag = (prices[cdates,:,:][:,:,positions]).min(axis = 2) > row.strike

        # купон с памятью
        coupon_memory = pd.DataFrame(np.zeros(coupon_flag.shape))
        for i in range(0,coupon_flag.shape[0]):
            coupon_memory.ix[i] = ((i+1) - coupon_memory.ix[:i].sum(axis = 0))*pd.DataFrame(coupon_flag).ix[i]
        
        coupon_i = row.coupon_i
        if ((index == 0) and (response_option == 1)):
            coupon_i = row.coupon_i*response_option_value
        
        cf_rub[cdates,:,index] = np.array(coupon_memory)*coupon_i/row.freq*(row.und_curr == 'RUB')
        cf_usd[cdates,:,index] = np.array(coupon_memory)*coupon_i/row.freq*(row.und_curr == 'USD')

    prods_rub_2 = prods_rub_2 + cf_rub
    prods_usd_2 = prods_usd_2 + cf_usd  

    return prods_rub_2, prods_usd_2, list(prods.name.values)

def calc_autocall_notes(prods, N_sims, Years, points_in_year, client_term, prices, symbols, response_option = 0, response_option_value = 0):

    ### достаем выкупные из отдельного файла, приводим массив к виду n_periods x n_prods
    n_periods = Years * points_in_year

    bs, buyouts = read_buyouts('buyoutscheme.xlsx', points_in_year, Years)
    positions_bs = []
    for a in prods.buyout:
        positions_bs.append(np.where(buyouts == a)[0][0])
    bs = bs[:,positions_bs]  

    ##срок - ровно срок продукта
    term_row = (prods.term * points_in_year - 1)
    term_table = np.zeros((n_periods, prods.shape[0]))

    for i in range(prods.shape[0]):
        term_table [int(term_row[i]), i] = 1

    ## купон
    cf_rub = np.zeros((n_periods, N_sims, prods.shape[0]))
    cf_usd = np.zeros((n_periods, N_sims, prods.shape[0]))

    ### для каждого продукта отдельно
    for index, row in prods.iterrows():   

        ## купонные даты
        term = min(row.term,client_term)
        cdates = range(int(points_in_year/row.freq)-1,points_in_year*term,int(points_in_year/row.freq) )         

        g1 = term_table[:,index]
        g2 = g1[cdates]
        g3 = np.repeat(g2[:, np.newaxis], N_sims, axis=1)

        ## underlying position:
        und = pd.DataFrame(row[list(row.drop('und_curr').filter(regex='und'))].index).dropna()[0].values
        positions = []
        for a in und:
            positions.append(np.where(symbols == a)[0][0])        

        # пробит страйк по худшему активу
        worst_stock_cdate = (prices[cdates,:,:][:,:,positions]).min(axis = 2)
        coupon_flag = worst_stock_cdate > row.coupon_level

        # купон с памятью
        coupon_memory = pd.DataFrame(np.zeros(coupon_flag.shape))
        for i in range(0,coupon_flag.shape[0]):
            coupon_memory.ix[i] = ((i+1) - coupon_memory.ix[:i].sum(axis = 0))*pd.DataFrame(coupon_flag).ix[i]

        call_flag = worst_stock_cdate > row.call_level    
        call1 = np.concatenate((np.zeros((1,call_flag.shape[1])),call_flag), axis = 0)
        call2 = np.cumsum(call1, axis = 0) > 0
        call4 = np.diff(call2, axis = 0) 

        protection_flag = worst_stock_cdate < row.protection_level    
        pr1 = np.concatenate((np.zeros((1,protection_flag.shape[1])),protection_flag), axis = 0)
        pr2 = np.cumsum(pr1, axis = 0) > 0
        pr4 = np.diff(pr2, axis = 0) 

        active_flag = (np.multiply( \
            np.logical_not(np.cumsum(call_flag, axis = 0)), \
            np.logical_not(np.cumsum(protection_flag, axis = 0)))) 

        active_flag = np. concatenate(( \
                                      np.ones((1, N_sims)) ,\
                                      active_flag[:-1,:]), axis = 0)

        cf = np.multiply((call4 + coupon_memory*row.coupon/row.freq + np.multiply(pr4, worst_stock_cdate) + g3), active_flag)    
        cf_rub[cdates,:,index] = cf*(row.und_curr == 'RUB')
        cf_usd[cdates,:,index] = cf*(row.und_curr == 'USD')

    prods_rub_3 = cf_rub
    prods_usd_3 = cf_usd

    return prods_rub_3, prods_usd_3, list(prods.name.values)

prod_dict = {
    'исж классический': calc_prod_ili_classic,
    'исж купонный': calc_prod_ili_coupon,
    'автоколл': calc_autocall_notes,
    'депозит': calc_prod_deposits,
    'ПИФ': calc_prod_pifpaf,
    'БСО': calc_prod_exchange_structured_bonds
}

def calc_prod(name):
    prod_dict[name]()

def calc_products2(prices, Years, points_in_year, client_term, response_option = 0, response_option_value = 0):

    n_periods = Years * points_in_year   
    N_sims = prices.shape[1]
    prod_cat, prod_types, BAs = read_prodcat('prodcat.xlsx')
    symbols = np.array(BAs)

    prods_rub_1, prods_usd_1, names_1 = calc_prod_ili_classic(prod_cat['исж классический'], N_sims, Years, points_in_year, client_term, prices, symbols)
    prods_rub_2, prods_usd_2, names_2 = calc_prod_ili_coupon(prod_cat['исж купонный'],      N_sims, Years, points_in_year, client_term, prices, symbols)
    prods_rub_3, prods_usd_3, names_3 = calc_autocall_notes(prod_cat['автоколл'],           N_sims, Years, points_in_year, client_term, prices, symbols)
    prods_rub_4, prods_usd_4, names_4 = calc_prod_deposits(prod_cat['депозит'],             N_sims, Years, points_in_year, client_term, prices, symbols)
    prods_rub_5, prods_usd_5, names_5 = calc_prod_pifpaf(prod_cat['ПИФ'],                   N_sims, Years, points_in_year, client_term, prices, symbols)
    prods_rub_6, prods_usd_6, names_6 = calc_prod_exchange_structured_bonds(prod_cat['БСО'],N_sims, Years, points_in_year, client_term, prices, symbols)
        
    prods_rub_portf = np.concatenate([prods_rub_1, prods_rub_2, prods_rub_3, prods_rub_4, prods_rub_5, prods_rub_6], axis = 2) 
    prods_usd_portf = np.concatenate([prods_usd_1, prods_usd_2, prods_usd_3, prods_usd_4, prods_usd_5, prods_usd_6], axis = 2) 
    
    total = np.concatenate([-np.ones((1, prods_usd_portf.shape[1], prods_usd_portf.shape[2])), prods_usd_portf], axis = 0)
    names = names_1 + names_2 + names_3 + names_4 + names_5 + names_6

    #print('done, my leatherbag')

    return total, names



