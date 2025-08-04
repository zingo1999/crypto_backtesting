
import os

import pandas as pd

from analysis_tools.backtest_engine import BacktestEngine
from analysis_tools.crypto_data_service import CryptoDataService
from analysis_tools.crypto_exchange_data_service import CryptoExchangeDataService
from analysis_tools.utilities import Utilities


class ParameterPlateau:
    def __init__(self, asset_currency, kwargs):
        self.asset_currency_list = ['BTC', 'ETH', 'SOL'] if not asset_currency else [asset_currency]
        self.kwargs = kwargs

    def optimize_parameters(self):
        for asset_currency in self.asset_currency_list:
            self.kwargs['asset_currency'] = asset_currency
            file_path = f"backtest_results/{asset_currency}/{asset_currency}_filtered_result.csv"
            if os.path.exists(file_path):
                result_df = pd.read_csv(file_path, index_col=0)
                result_df = result_df[result_df['sharpe'] > self.kwargs['minimum_sharpe']].sort_values(by='strategy').reset_index(drop=True)
                check_parameter_plateau = self.process_results(result_df, self.kwargs)
                pp_columns = (check_parameter_plateau.columns)[:-1]
                result_df = result_df.drop(columns=pp_columns, errors='ignore')

                result_df = pd.merge(result_df, check_parameter_plateau, on='strategy', how='inner')
                cols = [col for col in result_df.columns if col != 'strategy'] + ['strategy']
                result_df = result_df[cols]
                result_df = result_df.sort_values(by='sharpe', ascending=False).reset_index(drop=True)

                result_df.to_csv(file_path)
                print(result_df.head())
                pass

    @staticmethod
    def evaluate_strategy(task, **kwargs):
        x = task['x']
        y = task['y']
        lookback_list = [max(x - 10, 0), x, x + 10]
        threshold_list = task['threshold_list']
        threshold_list = list({max(y - (threshold_list[1] - threshold_list[0]), 0), y, min(threshold_list[-1], y + (threshold_list[1] - threshold_list[0]))})

        task.update({
            'lookback_list': lookback_list,
            'threshold_list': threshold_list,
        })
        performance_metrics = BacktestEngine.performance_evaluation(task)
        if performance_metrics:
            sharpe_ratios = []
            for i in range(len(performance_metrics)):
                sharpe_ratios.append(performance_metrics[i]['result']['sharpe'])
            parameter_plateau = all(x > 1 for x in sharpe_ratios)
            return {
                'parameter_plateau': parameter_plateau,
                'strategy': task['strategy'], }
        else: return {
            'parameter_plateau': False,
            'strategy': task['strategy'],
        }

    @staticmethod
    def process_results(result_df, kwargs):
        strategy_key_list = result_df['strategy'].str.rsplit('|', n=5).str[:-5].str.join('|').unique().tolist()
        backtest_dataframe_map = {strategy_key: CryptoDataService({**kwargs,
            'asset_currency': key_parts[1],
            'data_source': key_parts[3],
            'factor_currency': key_parts[0],
            'timeframe': key_parts[2],
            'endpoint': key_parts[-1]}).create_backtest_dataframe(CryptoExchangeDataService(**{**kwargs, 'timeframe': key_parts[2]}).get_historical_data(True)) for strategy_key in strategy_key_list for key_parts in [strategy_key.split('|')]}

        tasks = []
        for _, row in result_df.iterrows():
            backtest_dataframe_key = '|'.join(row['strategy'].rsplit('|', 5)[:-5])
            backtest_df = backtest_dataframe_map[backtest_dataframe_key].copy()

            threshold_list = Utilities.generate_threshold_list(backtest_df, row['strategy'].split('|')[-5])

            tasks.append({
                'action': row['strategy'].split('|')[-3],
                'asset_currency': row['strategy'].split('|')[1],
                'backtest_df': backtest_df,
                'data_source': row['strategy'].split('|')[3],
                'endpoint': row['strategy'].split('|')[4],
                'factor_currency': row['strategy'].split('|')[0],
                'indicator': row['strategy'].split('|')[-5],
                # 'lookback_list': lookback_list,
                # 'minimum_sharpe': kwargs['minimum_sharpe'],
                'orientation': row['strategy'].split('|')[-4],
                # 'sharpe': data['sharpe'],
                'strategy': row['strategy'],
                'threshold_list': threshold_list,
                'timeframe': row['strategy'].split('|')[2],
                'x': row['x'],
                'y': row['y'],
            })
        if tasks:
            evaluation_results = (Utilities.run_in_parallel(ParameterPlateau.evaluate_strategy, tasks) if len(tasks) > 1 else [ParameterPlateau.evaluate_strategy(tasks[0])])
            return pd.DataFrame(evaluation_results)