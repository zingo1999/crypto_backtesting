# Standard library imports
import pandas as pd

# Third-party imports



# Custom imports
from crypto_data_serivce import CryptoDataService
from crypto_exchange_data_service import CryptoExchangeDataService
from utilities import Utilities




class DataAnalysis:



    def __init__(self, kwargs):
        self.kwargs = kwargs
        # self.asset_currency = ''
        # self.factor_currency = ''
        # self.indicator = ''
        # self.timeframe = ''
        # for key, value in kwargs.items(): setattr(self, key, value)
        #
        # self.asset_currency_list = [self.asset_currency.upper()] if self.asset_currency else ['BTC', 'ETH', 'SOL']
        # self.factor_currency_list = [self.factor_currency.upper()] if self.factor_currency else ['BTC', 'ETH', 'SOL']
        # self.indicator_list = [self.indicator] if self.indicator else ['bband', 'rsi', 'ma_diff']
        # self.timeframe_list = [self.timeframe] if self.timeframe else ['1h', '1d']


    def data_analysis(self,):
        service = CryptoDataService(self.kwargs)
        backtest_combos = service.backtest_combinations()

        # for timeframe in self.timeframe_list:
        #     for asset_currency in self.asset_currency_list:
        #         price_df = CryptoExchangeDataService(asset_currency, **self.kwargs).get_historical_data(True)
        #         for factor_currency in self.factor_currency_list:
        #             self.kwargs.update({
        #                 'asset_currency': asset_currency,
        #                 'factor_currency': factor_currency,
        #                 'timeframe': timeframe,
        #             })
        #             backtest_df = CryptoDataService(self.kwargs).create_backtest_df(price_df.copy())
        #             print(f"{backtest_df.tail(1)}\n")
        #             all_lookback_lists = Utilities.generate_lookback_lists(backtest_df.copy())
        #             for indicator in self.indicator_list:
        #                 threshold_lists = Utilities.generate_threshold_list(backtest_df.copy(), indicator)
        #                 for orientation in orientation_list:
        #                     for action in action_list:

        pass

