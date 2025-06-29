import pandas as pd
import multiprocessing as mp

from crypto_exchange_data_service import CryptoExchangeDataService
from glassnode_data_service import GlassnodeDataService
from utilities import Utilities

class CryptoDataService:



    def __init__(self, kwargs):
        # self.asset_currency = ''
        # self.data_source = ''
        # self.endpoint = ''
        # self.factor_currency = ''
        # self.kwargs = kwargs
        # for key, value in kwargs.items(): setattr(self, key, value)

        self.kwargs = kwargs

        self.action = ''
        self.asset_currency = ''
        self.data_source = ''
        self.endpoint = ''
        self.factor_currency = ''
        self.indicator = ''
        self.orientation = ''
        self.timeframe = ''
        for key, value in kwargs.items(): setattr(self, key, value)

        self.action_list = [self.action] if self.action else ['long_short', 'long_only', 'short_only']
        self.asset_currency_list = [self.asset_currency.upper()] if self.asset_currency else ['BTC', 'ETH', 'SOL']
        self.factor_currency_list = [self.factor_currency.upper()] if self.factor_currency else ['BTC', 'ETH', 'SOL']
        self.indicator_list = [self.indicator] if self.indicator else ['bband', 'rsi', 'ma_diff']
        self.orientation_list = [self.orientation] if self.orientation else ['momentum', 'reversion']
        self.timeframe_list = [self.timeframe] if self.timeframe else ['1h', '1d']

    def create_factor_df(self):
        factor_df = pd.DataFrame()
        if self.data_source == 'exchange':
            # exchange_data = CryptoExchangeDataService(self.factor_currency, **self.kwargs)
            # factor_df = exchange_data.get_historical_data()
            factor_df = CryptoExchangeDataService(self.factor_currency, **self.kwargs).get_historical_data()

        elif self.data_source == 'glassnode':
            # gn_service = GlassnodeDataService(self.kwargs)
            # factor_df = gn_service.fetch_data()
            factor_df = GlassnodeDataService(self.kwargs).fetch_data()

        return factor_df

    def create_price_df(self):
        exchange_data = CryptoExchangeDataService(self.asset_currency, **self.kwargs)
        price_df = exchange_data.get_historical_data(True)
        return price_df

    def generate_endpoint_list(self):
        endpoint_list = []
        if self.data_source == 'exchange':
            endpoint_list = [self.endpoint]
        elif self.data_source == 'glassnode':
            endpoint_list = GlassnodeDataService(self.kwargs).get_endpoint_list()
        return endpoint_list

    def extract_values(self, row):
        data = row['o']
        extracted = {}

        for key in data.keys():
            if len(data) >= 2:
                extracted[key] = data[key]
        return pd.Series(extracted)

    def create_backtest_df(self, price_df=pd.DataFrame()):
        if price_df.empty: price_df = self.create_price_df()
        factor_df = self.create_factor_df()
        if factor_df.empty: return pd.DataFrame()

        if self.data_source == 'exchange':
            if self.endpoint in ['premium_index', 'price']:
                factor_df = factor_df[['unix_timestamp', 'close']].rename(columns={'close': 'factor'})
        elif self.data_source == 'glassnode':
            if factor_df.columns[-1] == 'o':
                extracted_df = factor_df.apply(self.extract_values, axis=1)
                factor_df = pd.concat([factor_df, extracted_df], axis=1)

                factor_df['average_value'] = factor_df.iloc[:, 2:].mean(axis=1)
                pass

            factor_df = factor_df.copy()
            factor_df.loc[:, 't'] = factor_df['t'] * 1000
            factor_df = factor_df[['t', factor_df.columns[-1]]]
            factor_df.rename(columns={'t': 'unix_timestamp', factor_df.columns[-1]: 'factor'}, inplace=True)

            pass
        price_df = price_df[['unix_timestamp', 'close']].rename(columns={'close': 'price'})

        backtest_df = pd.merge(factor_df, price_df, how='inner', on='unix_timestamp')
        backtest_df = backtest_df.assign(datetime=lambda x: pd.to_datetime(x['unix_timestamp'], unit='ms'))
        backtest_df = backtest_df.drop_duplicates(subset='unix_timestamp', keep='last').sort_values('unix_timestamp').set_index('datetime').dropna()
        return backtest_df

    def backtest_combinations(self,):
        all_results = []
        for timeframe in self.timeframe_list:
            for asset_currency in self.asset_currency_list:
                price_df = CryptoExchangeDataService(asset_currency, **self.kwargs).get_historical_data(True)
                endpoint_list = self.generate_endpoint_list()
                for endpoint in endpoint_list:
                    self.kwargs.update({'endpoint': endpoint})
                    for factor_currency in self.factor_currency_list:
                        self.kwargs.update({
                            'asset_currency': asset_currency,
                            'factor_currency': factor_currency,
                            'timeframe': timeframe,
                        })
                        backtest_df = CryptoDataService(self.kwargs).create_backtest_df(price_df.copy())
                        if backtest_df.empty:
                            continue
                        print(f"{backtest_df.tail(1)}\n")
                        all_lookback_lists = Utilities.generate_lookback_lists(backtest_df.copy())
                        backtest_combos = []
                        for indicator in self.indicator_list:
                            threshold_list = Utilities.generate_threshold_list(backtest_df.copy(), indicator)
                            # if indicator == 'roc': threshold_list = all_lookback_lists[0][:15]
                            # else: threshold_list = Utilities.generate_threshold_list(backtest_df.copy(), indicator)
                            for orientation in self.orientation_list:
                                for action in self.action_list:
                                    para_combination = {
                                        'all_lookback_lists': all_lookback_lists,
                                        'asset_currency': asset_currency,
                                        'df': backtest_df,
                                        # 'data_source': data_source,
                                        # 'endpoint_path': endpoint_path,
                                        'factor_currency': factor_currency,
                                        'indicator': indicator,
                                        # 'max_recovery_days': max_recovery_days,
                                        # 'minimum_sharpe': minimum_sharpe,
                                        # 'minimum_trades': minimum_trades,
                                        # 't_plus': t_plus,
                                        'threshold_list': threshold_list,
                                        'timeframe': timeframe,
                                        'title': f"{factor_currency}_{asset_currency}_{self.data_source}_{self.endpoint}_{timeframe}_{indicator}_{orientation}_{action}",
                                        'action': action,
                                        'orientation': orientation, }
                                    backtest_combos.append(para_combination)

                        num_cores = min(len(backtest_combos), 6)
                        pool = mp.Pool(processes=num_cores)
                        backtest_results = pool.map(Utilities.backtest_engine, backtest_combos)
                        pool.close()
                        backtest_results = [result for result in backtest_results if result is not None]
                        if backtest_results: all_results.append(backtest_results)


                        # if any(element is not None for element in backtest_results):
                        #     all_results.append(backtest_results)


        Utilities.generate_heatmap(all_results, True)
        pass