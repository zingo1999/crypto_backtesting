
import math
import os

import numpy as np
import pandas as pd


from backtest_engine import BacktestEngine
from crypto_data_service import CryptoDataService
from crypto_exchange_data_service import CryptoExchangeDataService
from utilities import Utilities

class WalkForwardAnalysis:
    def __init__(self, asset_currency, kwargs):
        self.asset_currency_list = ['BTC', 'ETH', 'SOL'] if not asset_currency else [asset_currency]
        self.kwargs = kwargs


    @staticmethod
    def evaluate_task_performance(task, **kwargs):
        if task['lookback_list'][0] > len(task['training_set']) or task['lookback_list'][0] > len(task['testing_set']):
            return {
                'training_set': np.nan,
                'testing_set': np.nan,
                'strategy': task['strategy'],
            }

        task.update({
        'backtest_df': task['training_set'],
        'period': 'in_sample',
    })
        training_eval = BacktestEngine.performance_evaluation(task)
        training_sharpe = training_eval[0]['result']['sharpe'] if training_eval else np.nan
        task.update({
            'backtest_df': task['testing_set'],
            'period': 'out_of_sample',
        })
        testing_eval = BacktestEngine.performance_evaluation(task)
        testing_sharpe = testing_eval[0]['result']['sharpe'] if testing_eval else np.nan
        summary = {
            'training_set': training_sharpe,
            'testing_set': testing_sharpe,
            'strategy': task['strategy'],
        }
        return summary

    @staticmethod
    def prepare_and_execute_walk_forward(result_df, kwargs):
        strategy_key_list = result_df['strategy'].str.rsplit('|', n=5).str[:-5].str.join('|').unique().tolist()
        backtest_dataframe_map = {strategy_key: CryptoDataService({
            **kwargs,
            'asset_currency': key_parts[1],
            'data_source': key_parts[3],
            'factor_currency': key_parts[0],
            'timeframe': key_parts[2],
            'endpoint': key_parts[-1]
        }).create_backtest_dataframe(CryptoExchangeDataService(**{**kwargs, 'timeframe': key_parts[2]}).get_historical_data(True)) for strategy_key in strategy_key_list for key_parts in [strategy_key.split('|')]}

        tasks = []
        for i in range(len(result_df)):
            row = result_df.iloc[i]
            backtest_dataframe_key = '|'.join(row['strategy'].rsplit('|', 5)[:-5])
            backtest_df = backtest_dataframe_map[backtest_dataframe_key].copy()

            train_size = math.floor(len(backtest_df) * 0.7)
            training_set = backtest_df.copy().iloc[:train_size]
            testing_set = backtest_df.copy().iloc[train_size:]

            tasks.append({
                'action': row['strategy'].split('|')[-3],
                'asset_currency': row['strategy'].split('|')[1],
                'data_source': row['strategy'].split('|')[3],
                'endpoint': row['strategy'].split('|')[4],
                'factor_currency': row['strategy'].split('|')[0],
                'indicator': row['strategy'].split('|')[-5],
                'lookback_list': [row['x']],
                # 'minimum_sharpe': kwargs['minimum_sharpe'],
                'orientation': row['strategy'].split('|')[-4],
                # 'sharpe': data['sharpe'],
                'strategy': row['strategy'],
                'testing_set': testing_set,
                'threshold_list': [row['y']],
                'timeframe': row['strategy'].split('|')[2],
                'training_set': training_set,
                # 'x': data['x'],
                # 'y': data['y'],
            })

        if tasks:
            walk_forward_results = (Utilities.run_in_parallel(WalkForwardAnalysis.evaluate_task_performance, tasks) if len(tasks) > 1 else [WalkForwardAnalysis.evaluate_task_performance(tasks[0])])
            return pd.DataFrame(walk_forward_results)

    def run_walk_forward_analysis(self):
        for asset_currency in self.asset_currency_list:
            self.kwargs['asset_currency'] = asset_currency
            file_path = f"backtest_results/{asset_currency}/{asset_currency}_filtered_result.csv"
            if os.path.exists(file_path):
                result_df = pd.read_csv(file_path, index_col=0)
                result_df = result_df[result_df['sharpe'] > self.kwargs['minimum_sharpe']].sort_values(by='strategy').reset_index(drop=True)
                walk_forward_results = self.prepare_and_execute_walk_forward(result_df, self.kwargs)
                wf_columns = (walk_forward_results.columns)[:-1]
                result_df = result_df.drop(columns=wf_columns, errors='ignore')

                result_df = pd.merge(result_df, walk_forward_results, on='strategy', how='inner')
                cols = [col for col in result_df.columns if col != 'strategy'] + ['strategy']
                result_df = result_df[cols]
                result_df = result_df.sort_values(by='sharpe', ascending=False).reset_index(drop=True)

                # file_path = f"backtest_results/{asset_currency}/{asset_currency}_walk_forward_result.csv"
                result_df.to_csv(file_path)
                print(result_df.head())
                pass

