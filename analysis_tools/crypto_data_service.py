import pandas as pd
import os

from analysis_tools.backtest_engine import BacktestEngine
from analysis_tools.crypto_exchange_data_service import CryptoExchangeDataService
from analysis_tools.glassnode_data_service import GlassnodeDataService
from analysis_tools.utilities import Utilities


class CryptoDataService:

    def __init__(self, params: dict):
        """
        初始化 CryptoDataService 實例。

        :param params: 包含所有初始化參數的字典
        """
        # 設置默認值
        self.action = params.get('action', '')
        self.asset_currency = params.get('asset_currency', '')
        self.cross_validate = params.get('cross_validate', '')
        self.data_source = params.get('data_source', '')
        self.endpoint = params.get('endpoint', '')
        self.factor_currency = params.get('factor_currency', '')
        self.indicator = params.get('indicator', '')
        self.kwargs = params
        self.max_threshold = params.get('max_threshold', None)
        self.minimum_sharpe = params.get('minimum_sharpe', None)
        self.number_of_interval = params.get('number_of_interval', None)
        self.orientation = params.get('orientation', '')
        self.timeframe = params.get('timeframe', '')
        self.walk_forward = params.get('walk_forward', False)
        self.x = params.get('x', None)
        self.y = params.get('y', None)

        # 初始化其他屬性
        self.action_list = [self.action] if self.action else ['long_short', 'long_only', 'short_only']
        self.asset_currency_list = [self.asset_currency.upper()] if self.asset_currency else ['BTC', 'ETH', 'SOL']
        self.factor_currency_list = [self.factor_currency.upper()] if self.factor_currency else ['BTC', 'ETH', 'SOL']
        self.indicator_list = [self.indicator] if self.indicator else ['bband', 'rsi']
        self.orientation_list = [self.orientation] if self.orientation else ['momentum', 'reversion']
        self.timeframe_list = [self.timeframe] if self.timeframe else ['24h', '12h', '1h']  # ['10m', '1h', '1d']
        self.lookback_list = [self.x] if self.x else []  # ['10m', '1h', '1d']
        self.threshold_list = [self.y] if self.y else []  # ['10m', '1h', '1d']

    def get_factor_dataframe(self,):
        factor_df = pd.DataFrame()
        if self.data_source == 'exchange':
            kwargs = self.kwargs.copy()
            kwargs['asset_currency'] = self.factor_currency
            factor_df = CryptoExchangeDataService(**kwargs).get_historical_data()
        elif self.data_source == 'glassnode':
            factor_df = GlassnodeDataService(self.kwargs).fetch_data()
        return factor_df

    def get_price_dataframe(self,):
        exchange_data = CryptoExchangeDataService(**self.kwargs)
        price_df = exchange_data.get_historical_data(True)
        return price_df

    def get_endpoint_list(self):
        endpoint_list = []
        if self.data_source == 'exchange':
            endpoint_list = [self.endpoint]
        elif self.data_source == 'glassnode':
            endpoint_list = GlassnodeDataService(self.kwargs).get_endpoint_list()
        return endpoint_list

    @staticmethod
    def extract_nested_data(factor_df):
        extracted_df = pd.json_normalize(factor_df['o'])
        return extracted_df

    def create_backtest_dataframe(self, price_df=pd.DataFrame()):
        if price_df.empty:
            price_df = self.get_price_dataframe()

        factor_df = self.get_factor_dataframe()
        if factor_df is None or factor_df.empty:
            return pd.DataFrame()

        if self.data_source == 'exchange' and self.endpoint in ['premium_index', 'price']:
            factor_df = factor_df[['unix_timestamp', 'close']].rename(columns={'close': 'factor'})
        elif self.data_source == 'glassnode':
            if factor_df.columns[-1] == 'o':
                extracted_df = self.extract_nested_data(factor_df)
                factor_df = pd.concat([factor_df, extracted_df], axis=1)
                factor_df['average_value'] = factor_df.iloc[:, 2:].mean(axis=1)

            factor_df['t'] *= 1000
            factor_df = factor_df[['t', factor_df.columns[-1]]].rename(columns={'t': 'unix_timestamp', factor_df.columns[-1]: 'factor'})

        price_df = price_df[['unix_timestamp', 'close']].rename(columns={'close': 'price'})

        try:
            backtest_df = pd.merge(factor_df, price_df, how='inner', on='unix_timestamp')
        except Exception as e:
            print(f"{e}")
            return

        backtest_df['datetime'] = pd.to_datetime(backtest_df['unix_timestamp'], unit='ms')
        backtest_df = backtest_df.drop_duplicates(subset='unix_timestamp', keep='last').set_index('datetime')
        backtest_df['chg'] = backtest_df['price'].pct_change()

        return backtest_df

    def generate_all_backtest_results(self, ):
        all_results = {}
        for asset_currency in self.asset_currency_list:
            for timeframe in self.timeframe_list:
                self.kwargs.update({
                    'asset_currency': asset_currency,
                    'timeframe': timeframe,
                })
                price_df = CryptoExchangeDataService(**self.kwargs).get_historical_data(True)
                endpoint_list = self.get_endpoint_list()
                for endpoint in endpoint_list:
                    self.kwargs.update({'endpoint': endpoint})
                    for factor_currency in self.factor_currency_list:
                        self.kwargs.update({'factor_currency': factor_currency})
                        backtest_df = CryptoDataService(self.kwargs).create_backtest_dataframe(price_df.copy())
                        if backtest_df.empty: continue
                        print(f"{backtest_df.tail(2)}\n")
                        lookback_list = Utilities.generate_lookback_lists(backtest_df.copy()) if not self.lookback_list else self.lookback_list
                        backtest_combos = []
                        for indicator in self.indicator_list:
                            threshold_list = Utilities.generate_threshold_list(backtest_df.copy(), indicator, self.max_threshold, self.number_of_interval) if not self.threshold_list else self.threshold_list
                            for orientation in self.orientation_list:
                                for action in self.action_list:
                                    para_combination = {
                                        'action': action,
                                        'asset_currency': asset_currency,
                                        'backtest_df': backtest_df,
                                        'data_source': self.data_source,
                                        'endpoint': endpoint,
                                        'factor_currency': factor_currency,
                                        'indicator': indicator,
                                        'lookback_list': lookback_list,
                                        'orientation': orientation,
                                        'minimum_sharpe': self.kwargs['minimum_sharpe'],
                                        # 't_plus': t_plus,
                                        'strategy': f"{indicator}{orientation}{action}",
                                        'threshold_list': threshold_list,
                                        'timeframe': timeframe,
                                        'title': f"{factor_currency}{asset_currency}{self.data_source}{endpoint}{timeframe}{indicator}{orientation}_{action}",
                                    }
                                    backtest_combos.append(para_combination)
                        del lookback_list, threshold_list
                        if len(backtest_combos) > 1: backtest_results = Utilities.run_in_parallel(BacktestEngine.performance_evaluation, backtest_combos)
                        else: backtest_results = [BacktestEngine.performance_evaluation(backtest_combos[0])]
                        if any(element is not None for element in backtest_results):
                            backtest_results = [result for result in backtest_results if result is not None]
                            if asset_currency not in all_results: all_results[asset_currency] = []
                            all_results[asset_currency].append(backtest_results)
                            del backtest_results
        if all_results:
            asset_currency_keys = list(all_results.keys())
            for asset_currency in asset_currency_keys:
                backtest_results_folder = f"backtest_results/{asset_currency}/{self.kwargs['since']}"
                os.makedirs(backtest_results_folder, exist_ok=True)
                file_path = os.path.join(backtest_results_folder, f"{asset_currency}_backtest_result")
                different_timeframe_results = all_results[asset_currency]
                extracted_results = []
                for different_stratgy_results in different_timeframe_results:
                    for strategy_data_point in different_stratgy_results:
                        for data in strategy_data_point:
                            extracted_results.append(data['result'])
                results_df = pd.DataFrame(extracted_results)

                file_name = f"{file_path}.csv"
                if os.path.exists(file_name):
                    existing_df = pd.read_csv(file_name, index_col=0)
                    results_df = pd.concat([existing_df, results_df], ignore_index=True).drop_duplicates(subset='strategy', keep='last')

                results_df = results_df[results_df['sharpe'] >= self.minimum_sharpe].sort_values(by='sharpe', ascending=False).reset_index(drop=True)
                results_df.to_csv(file_name)
                print(results_df.head())
        return all_results

