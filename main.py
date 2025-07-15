
# Custom imports
from data_analysis import DataAnalysis


if __name__ == '__main__':

    factor_currency = 'btc'
    asset_currency  = 'btc'
    action          = ''
    indicator       = 'roc_hv'
    orientation     = ''
    since           = '2020-05-01'
    # since           = '2025-06-01'
    timeframe       = '1h'

    data_source     = 'exchange'        # glassnode
    endpoint        = 'price'       # active_1m_3m options_25delta_skew_3_months implied options_open_interest_distribution
    exchange_name   = 'bybit'
    product_type    = 'linear'

    update_mode         = True
    minimum_sharpe      = 0.5

    max_threshold       = 0
    number_of_interval  = 0

    cross_validate = False

    kwargs = {key: value for key, value in locals().items() if not key.startswith('__') and isinstance(value, (str, int, float, bool))}


    data_service = DataAnalysis(kwargs)
    data_service.data_analysis()



    ### fetch_mark_ohlcv                  # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_index_ohlcv                 # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_ohlcv                       # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_premium_index_ohlcv         # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_funding_rate_history        # only support linear and inverse
    ### fetch_open_interest_history       # only support linear and inverse
    ### fetch_long_short_ratio_history    # only support linear and inverse