import os

import numpy as np
import pandas as pd


from analysis_tools.backtest_engine import BacktestEngine
from analysis_tools.crypto_data_service import CryptoDataService
from analysis_tools.crypto_exchange_data_service import CryptoExchangeDataService
from analysis_tools.utilities import Utilities

#
# class ResultsFiltering:
#
#     def __init__(self, asset_currency, kwargs):
#         self.asset_currency_list = ['BTC', 'ETH', 'SOL'] if not asset_currency else [asset_currency]
#         self.kwargs = kwargs
#
#     def process_backtest_results(self,):
#         for asset_currency in self.asset_currency_list:
#             self.kwargs['asset_currency'] = asset_currency
#             file_path = f"backtest_results/{asset_currency}/{self.kwargs['since']}/{asset_currency}_filtered_result.csv"
#             if os.path.exists(file_path):
#                 result_df = pd.read_csv(file_path, index_col=0)
#                 result_df = result_df[(result_df['sharpe'] > self.kwargs['minimum_sharpe']) & (result_df['calmar'] >= 1) & (result_df['sharpe'] >= result_df['benchmark'] + 0.2)].reset_index(drop=True)
#                 temp_var1 = self.function_a(result_df, self.kwargs, self.kwargs['t_plus'])
#                 for n in temp_var1:
#                     temp_var7 = pd.DataFrame(n)
#                     temp_var8 = temp_var7.columns[:-1]
#                     result_df = result_df.drop(columns=temp_var8, errors='ignore')
#
#                     result_df = pd.merge(result_df, temp_var7, on='strategy', how='inner')
#                     cols = [col for col in result_df.columns if col != 'strategy'] + ['strategy']
#                     result_df = result_df[cols]
#                 result_df = result_df.dropna().sort_values(by='sharpe', ascending=False).reset_index(drop=True)
#                 result_df.to_csv(file_path)
#
#
#
    # @staticmethod
    # def function_a(result_df, kwargs, t_plus=None):
    #     strategy_key_list = result_df['strategy'].str.rsplit('|', n=5).str[:-5].str.join('|').unique().tolist()
    #     backtest_dataframe_map = {strategy_key: CryptoDataService({
    #         **kwargs,
    #         'asset_currency': key_parts[1],
    #         'data_source': key_parts[3],
    #         'factor_currency': key_parts[0],
    #         'timeframe': key_parts[2],
    #         'endpoint': key_parts[-1]
    #     }).create_backtest_dataframe(CryptoExchangeDataService(**{**kwargs, 'timeframe': key_parts[2]}).get_historical_data(True)) for strategy_key in strategy_key_list for key_parts in [strategy_key.split('|')]}
    #
    #     temp_var2 = []
    #     delay = [1, 3, 5] if not t_plus else [t_plus]
    #     for d in delay:
    #         tasks = []
    #         for _, row in result_df.iterrows():
    #             backtest_dataframe_key = '|'.join(row['strategy'].rsplit('|', 5)[:-5])
    #             backtest_df = backtest_dataframe_map[backtest_dataframe_key].copy()
    #             tasks.append({
    #                 'action': row['strategy'].split('|')[-3],
    #                 'asset_currency': row['strategy'].split('|')[1],
    #                 'backtest_df': backtest_df,
    #                 'data_source': row['strategy'].split('|')[3],
    #                 'endpoint': row['strategy'].split('|')[4],
    #                 'factor_currency': row['strategy'].split('|')[0],
    #                 'indicator': row['strategy'].split('|')[-5],
    #                 'lookback_list': [int(row['strategy'].split('|')[-2])],
    #                 'orientation': row['strategy'].split('|')[-4],
    #                 'strategy': row['strategy'],
    #                 't_plus': d,
    #                 'threshold_list': [float(row['strategy'].split('|')[-1])],
    #                 'timeframe': row['strategy'].split('|')[2],
    #             })
    #         if tasks:
    #             temp_var4 = Utilities.run_in_parallel(ResultsFiltering.function_b, tasks) if len(tasks) > 1 else [ResultsFiltering.function_b(tasks[0])]
    #             temp_var2.append(temp_var4)
    #
    #     return temp_var2
    #
    #
    # @staticmethod
    # def function_b(task, **kwargs):
    #     temp_var3 = BacktestEngine.performance_evaluation(task)
    #     sharpe_ratio = temp_var3[0]['result']['sharpe'] if temp_var3 else np.nan
    #     return {
    #         f"T+{task['t_plus']}": sharpe_ratio,
    #         'strategy': task['strategy'],
    #     }


class DelayedExecutionEvaluator:

    def __init__(self, asset_currency: str, kwargs: dict):
        self.asset_currency_list = ['BTC', 'ETH', 'SOL'] if not asset_currency else [asset_currency]
        self.kwargs = kwargs
        self.kwargs['update_mode'] = False

    def filter_and_process_backtest_results(self):
        for asset_currency in self.asset_currency_list:
            self.kwargs['asset_currency'] = asset_currency
            file_path = os.path.join("backtest_results", asset_currency, str(self.kwargs.get('since')), f"{asset_currency}_filtered_result.csv")
            if os.path.exists(file_path):
                result_df = pd.read_csv(file_path, index_col=0)
                result_df = result_df[result_df.apply(self.is_valid_backtest, axis=1)].reset_index(drop=True)

                delay_evaluation_results = self.evaluate_strategies_with_delay(result_df, self.kwargs, self.kwargs['t_plus'])
                for result_group in delay_evaluation_results:
                    evaluation_dataframe = pd.DataFrame(result_group)
                    result_df = self.merge_backtest_metrics(result_df, evaluation_dataframe)

                cols = [col for col in result_df.columns if col != 'strategy'] + ['strategy']
                result_df = result_df[cols]
                result_df = result_df.dropna(subset=['T+1', 'T+3']).sort_values(by='sharpe', ascending=False).reset_index(drop=True)
                result_df.to_csv(file_path)
                print(result_df.head())

    def is_valid_backtest(self, row):
        return (row['sharpe'] > self.kwargs['minimum_sharpe'] and
                row['calmar'] >= 1 and
                row['sharpe'] >= row['benchmark'] + 0.2)

    @staticmethod
    def merge_backtest_metrics(base_df, evaluation_df):
        cols_to_drop = evaluation_df.columns.difference(['strategy'])
        base_df = base_df.drop(columns=cols_to_drop, errors='ignore')
        return pd.merge(base_df, evaluation_df, on='strategy', how='inner')

    @staticmethod
    def evaluate_strategies_with_delay(result_df, kwargs, t_plus=None):
        strategy_key_list = result_df['strategy'].str.rsplit('|', n=5).str[:-5].str.join('|').unique().tolist()
        backtest_dataframe_map = {strategy_key: CryptoDataService({
            **kwargs,
            'asset_currency': key_parts[1],
            'data_source': key_parts[3],
            'factor_currency': key_parts[0],
            'timeframe': key_parts[2],
            'endpoint': key_parts[-1]
        }).create_backtest_dataframe(CryptoExchangeDataService(**{**kwargs, 'timeframe': key_parts[2]}).get_historical_data(True)) for strategy_key in strategy_key_list for key_parts in [strategy_key.split('|')]}

        all_results = []
        delay = [1, 3, 5] if not t_plus else [t_plus]
        for d in delay:
            tasks = []
            for _, row in result_df.iterrows():
                backtest_dataframe_key = '|'.join(row['strategy'].rsplit('|', 5)[:-5])
                backtest_df = backtest_dataframe_map[backtest_dataframe_key].copy()
                tasks.append({
                    'action': row['strategy'].split('|')[-3],
                    'asset_currency': row['strategy'].split('|')[1],
                    'backtest_df': backtest_df,
                    'data_source': row['strategy'].split('|')[3],
                    'endpoint': row['strategy'].split('|')[4],
                    'factor_currency': row['strategy'].split('|')[0],
                    'indicator': row['strategy'].split('|')[-5],
                    'lookback_list': [int(row['strategy'].split('|')[-2])],
                    'orientation': row['strategy'].split('|')[-4],
                    'strategy': row['strategy'],
                    't_plus': d,
                    'threshold_list': [float(row['strategy'].split('|')[-1])],
                    'timeframe': row['strategy'].split('|')[2], })
            if tasks:
                batch_result = Utilities.run_in_parallel(DelayedExecutionEvaluator.compute_strategy_sharpe, tasks) if len(tasks) > 1 else [
                    DelayedExecutionEvaluator.compute_strategy_sharpe(tasks[0])]
                all_results.append(batch_result)

        return all_results

    @staticmethod
    def compute_strategy_sharpe(task, **kwargs):
        result = BacktestEngine.performance_evaluation(task)
        sharpe_ratio = result[0]['result']['sharpe'] if result else np.nan
        return {
            f"T+{task['t_plus']}": sharpe_ratio,
            'strategy': task['strategy'],
        }