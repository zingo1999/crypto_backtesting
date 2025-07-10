
# Custom imports
from data_analysis import DataAnalysis
from crypto_exchange_data_service import CryptoExchangeDataService

if __name__ == '__main__':

    factor_currency = ''
    asset_currency  = 'btc'
    action          = ''
    indicator       = ''
    orientation     = 'momentum'
    since           = '2020-05-01'
    # since           = '2025-06-01'
    timeframe       = ''

    data_source     = 'glassnode'        # glassnode
    endpoint        = 'options_25delta_skew_3_months'       # active_1m_3m options_25delta_skew_3_months implied
    exchange_name   = 'bybit'
    product_type    = 'linear'

    update_mode         = True
    minimum_sharpe      = 1.2

    max_threshold       = 0
    number_of_interval  = 0

    cross_validate = True

    kwargs = {key: value for key, value in locals().items() if not key.startswith('__') and isinstance(value, (str, int, float, bool))}


    data_service = DataAnalysis(kwargs)
    data_service.data_analysis()

    ##### class 1 #####
    crypto_exchange_data_service = CryptoExchangeDataService(**kwargs)
    factor_df = crypto_exchange_data_service.get_historical_data()
    # price_df = crypto_exchange_data_service.get_historical_data(True)  # True = underlying asset 的價格數據

    print(factor_df.tail(3))
    # print(price_df.tail(3))

    # sys.exit()
    #
    # ##### class 2 #####
    # factor_df2 = ConvertCcxtDataSerice.get_historical_data(**kwargs)
    # price_df2 = ConvertCcxtDataSerice.get_historical_data(underlying_asset=True, **kwargs)



    ### fetch_mark_ohlcv                  # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_index_ohlcv                 # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_ohlcv                       # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_premium_index_ohlcv         # spot - BTC/USDT, futures = BTC/USDT:USDT
    ### fetch_funding_rate_history        # only support linear and inverse
    ### fetch_open_interest_history       # only support linear and inverse
    ### fetch_long_short_ratio_history    # only support linear and inverse