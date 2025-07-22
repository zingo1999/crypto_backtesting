
# Custom imports
from data_analysis import DataAnalysis


if __name__ == '__main__':

    factor_currency = 'btc'
    asset_currency  = 'btc'
    action          = 'long_only'
    indicator       = 'bband'
    orientation     = ''
    since           = '2020-05-01'
    # since           = '2025-06-01'
    timeframe       = ''

    data_source     = 'glassnode'        # glassnode exchange
    endpoint        = 'options_25delta_skew_1'       # active_1m_3m options_25delta_skew_3_months implied options_open_interest_distribution
    exchange_name   = 'binance'
    product_type    = 'linear'

    max_threshold       = 0
    number_of_interval  = 0

    update_mode         = True

    minimum_sharpe      = 1.3
    position_count      = False
    walk_forward        = False
    cross_validate      = True

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