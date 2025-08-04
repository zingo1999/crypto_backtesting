
# Custom imports
from analysis_tools.data_analysis import DataAnalysis


factor_currency = 'btc'
asset_currency  = 'btc'
action          = 'long_short'  # long_short/long_only/short_only
indicator       = ''    # bband/rsi
orientation     = ''    # momentum/reversion
since           = '2020-05-01'  # 2020-05-01
timeframe       = ''       # 24h/12h/6h/3h/1h/30m/15m/10m/5m/3m
x               = 0
y               = 0

data_source     = 'exchange'  # glassnode exchange
endpoint        = 'price'  # active_1m_3m options_25delta_skew_3_months implied options_open_interest_distribution
exchange_name   = 'binance'
product_type    = 'linear'

max_threshold       = 0
number_of_interval  = 0

update_mode     = True
minimum_sharpe  = 1
position_count  = False

##### Mode #####
backtest_mode       = True
cross_validate      = True
parameter_plateau   = True
walk_forward        = True

generate_equity_curve   = True
show_equity_curve       = False

show_heatmap        = False
target_metric       = 'sharpe'  # sharpe, mdd, calmar,

dash_board = False

specific_task = ''


asset_currency, factor_currency = map(lambda currency: str(currency).upper(), (asset_currency, factor_currency))
kwargs = {key: value for key, value in locals().items() if not key.startswith('__') and isinstance(value, (str, int, float, bool))}

if __name__ == '__main__':

    data_service = DataAnalysis(kwargs)
    data_service.data_analysis()



    ### fetch_mark_ohlcv                  # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_index_ohlcv                 # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_ohlcv                       # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_premium_index_ohlcv         # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_funding_rate_history        # only support linear and inverse
    ### fetch_open_interest_history       # only support linear and inverse
    ### fetch_long_short_ratio_history    # only support linear and inverse