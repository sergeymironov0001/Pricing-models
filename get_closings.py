def get_closings(stock_list):

    import pandas as pd

    closing_all = pd.read_hdf('market_data_panel.h5', 'key').Close

    # load stock prices and risk free    
    panel = pd.read_hdf('market_data_panel.h5', 'key').swapaxes('items', 'minor')
    
    symbols = list(panel.items)
    n_assets = len(symbols)
    
    symbols2add = list(set(stock_list) - set(symbols))
    
    if len(symbols2add) > 0:
        
        news = DataReader(symbols2add, "google", pause=1).swapaxes('items', 'minor')
        panel = pd.concat([news, panel])
    
    panel.swapaxes('items', 'minor').to_hdf('market_data_panel.h5', 'key')

    return panel[stock_list].swapaxes('items', 'minor').Close