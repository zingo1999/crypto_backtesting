import os

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from analysis_tools.backtest_engine import BacktestEngine
from analysis_tools.crypto_data_service import CryptoDataService
from analysis_tools.crypto_exchange_data_service import CryptoExchangeDataService
from analysis_tools.utilities import Utilities


class HeatmapGenerator:

    def __init__(self, asset_currency, kwargs):
        self.asset_currency_list = ['BTC', 'ETH', 'SOL'] if not asset_currency else [asset_currency]
        self.kwargs = kwargs

    def generate_all_heatmaps(self, all_results, show_heatmap=False, target_metric='sharpe'):
        if not all_results: all_results = self.collect_backtest_results()
        asset_currency_list = list(all_results.keys())
        self.dispatch_heatmap_generation(all_results, asset_currency_list, show_heatmap, target_metric)

    def collect_backtest_results(self, ):
        all_results = {}
        for asset_currency in self.asset_currency_list:
            self.kwargs['asset_currency'] = asset_currency
            file_path = f"backtest_results/{asset_currency}/{asset_currency}_filtered_result.csv"
            if os.path.exists(file_path):
                result_df = pd.read_csv(file_path, index_col=0)
                result_df = result_df[result_df['sharpe'] > self.kwargs['minimum_sharpe']].sort_values(by='strategy').reset_index(drop=True)
                backtest_task_results = self.prepare_backtest_tasks(result_df, self.kwargs)
                all_results.update({asset_currency: backtest_task_results})
        return all_results

    @staticmethod
    def dispatch_heatmap_generation(all_results, asset_currency_list, show_heatmap, target_metric):
        for asset_currency in asset_currency_list:
            backtest_results_folder = f"backtest_results/{asset_currency}"
            os.makedirs(backtest_results_folder, exist_ok=True)
            different_timeframe_results = all_results[asset_currency]
            tasks = []
            for results in different_timeframe_results:
                if len(results) > 1: show_heatmap = False
                for heatmap in results:
                    tasks.append({
                        'backtest_results_folder': backtest_results_folder,
                        'heatmap': heatmap,
                        'generate_equity_curve': False,
                        'show_heatmap': show_heatmap,
                        'target_metric': target_metric,
                    })
                Utilities.run_in_parallel(HeatmapGenerator.generate_heatmap_from_task, tasks) if len(tasks) > 1 else HeatmapGenerator.generate_heatmap_from_task(tasks[0])

    @staticmethod
    def prepare_backtest_tasks(result_df, kwargs):
        dataframe_key_list = result_df['strategy'].str.rsplit('|', n=5).str[:-5].str.join('|').unique().tolist()
        backtest_dataframe_map = {strategy_key: CryptoDataService({
            **kwargs,
            'asset_currency': key_parts[1],
            'data_source': key_parts[3],
            'factor_currency': key_parts[0],
            'timeframe': key_parts[2],
            'endpoint': key_parts[-1]
        }).create_backtest_dataframe(CryptoExchangeDataService(**{**kwargs, 'timeframe': key_parts[2]}).get_historical_data(True)) for strategy_key in dataframe_key_list for key_parts in [strategy_key.split('|')]}

        heatmap_key_list = result_df['strategy'].str.rsplit('|', n=5).str[:-2].str.join('|').unique().tolist()
        timeframe_list = list({strategy_key.split('|')[2] for strategy_key in dataframe_key_list})

        heatmap_tasks_results = []
        for timeframe in timeframe_list:
            tasks = []
            for heatmap in heatmap_key_list:
                backtest_dataframe_key = '|'.join(heatmap.rsplit('|', 5)[:-3])
                backtest_df = backtest_dataframe_map[backtest_dataframe_key].copy()
                lookback_list = Utilities.generate_lookback_lists(backtest_df)
                threshold_list = Utilities.generate_threshold_list(backtest_df, heatmap.split('|')[-3])
                if heatmap.split('|')[2] == timeframe: tasks.append({
                    'action': heatmap.split('|')[-1],
                    'asset_currency': heatmap.split('|')[1],
                    'backtest_df': backtest_df,
                    'data_source': heatmap.split('|')[3],
                    'endpoint': heatmap.split('|')[4],
                    'factor_currency': heatmap.split('|')[0],
                    'indicator': heatmap.split('|')[-3],
                    'lookback_list': lookback_list,
                    'orientation': heatmap.split('|')[-2],
                    'threshold_list': threshold_list,
                    'timeframe': heatmap.split('|')[2],
                })
            if tasks:
                backtest_task_results = Utilities.run_in_parallel(HeatmapGenerator.evaluate_backtest_task, tasks) if len(tasks) > 1 else [HeatmapGenerator.evaluate_backtest_task(tasks[0])]
                heatmap_tasks_results.append(backtest_task_results)
        return heatmap_tasks_results

    @staticmethod
    def generate_heatmap_from_task(task):
        backtest_results_folder = task['backtest_results_folder']
        heatmap = task['heatmap']
        show_heatmap = task['show_heatmap']
        target_metric = task['target_metric']

        heatmap_data = []
        title = heatmap[0]['title']
        subfolder_path = os.path.join(backtest_results_folder, f"{heatmap[0]['action']}/{heatmap[0]['timeframe']}/{title.rsplit('|', 1)[0]}")

        for result in heatmap: heatmap_data.append(result['result'])

        chunks = [heatmap_data[i:i + 275] for i in range(0, len(heatmap_data), 275)]

        if len(chunks) >= 2:  # 雙圖模式
            if len(chunks) % 2 != 0: chunks.pop()
            for i in range(0, len(chunks), 2):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10))
                data_table1 = pd.DataFrame(chunks[i]).pivot_table(index='x', columns='y', values=target_metric)
                sns.heatmap(data_table1, ax=ax1, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})
                ax1.yaxis.set_tick_params(rotation=0)

                if i + 1 < len(chunks):
                    plt.suptitle(f"{title}\nFrom: {heatmap[0]['since']} to {heatmap[0]['end']}, Data quantity: {heatmap[0]['data_quantity']}")
                    data_table2 = pd.DataFrame(chunks[i + 1]).pivot_table(index='x', columns='y', values=target_metric)
                    sns.heatmap(data_table2, ax=ax2, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})
                    ax2.yaxis.set_tick_params(rotation=0)
                if show_heatmap:
                    plt.show()
                else:
                    file_path = os.path.join(subfolder_path, f"{title}_file{i + 1}")
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    fig.savefig(f"{file_path}_{target_metric}")
                plt.close(fig)
        else:  # 單圖模式
            plt.suptitle(f"{title}\nFrom: {heatmap[0]['since']} to {heatmap[0]['end']}, Data quantity: {heatmap[0]['data_quantity']}")
            fig, ax = plt.subplots(figsize=(10, 7))
            data_table = pd.DataFrame(chunks[0]).pivot_table(index='x', columns='y', values=target_metric)
            sns.heatmap(data_table, ax=ax, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})
            ax.yaxis.set_tick_params(rotation=0)
            if show_heatmap:
                plt.show()
            else:
                heatmap_file_path = os.path.join(subfolder_path, f"{title}_file1")
                fig.savefig(f"{heatmap_file_path}_{target_metric}")
            plt.close(fig)

    @staticmethod
    def evaluate_backtest_task(task, **kwargs):
        return BacktestEngine.performance_evaluation(task)


class EquityCurveGenerator:

    def __init__(self, asset_currency, kwargs):
        self.asset_currency_list = ['BTC', 'ETH', 'SOL'] if not asset_currency else [asset_currency]
        self.kwargs = kwargs

    def generate_equity_curves(self, equity_curve_data=False):
        all_results = self.load_all_results()
        asset_currency_list = list(all_results.keys())
        self.process_results(all_results, asset_currency_list, equity_curve_data)


    def load_all_results(self, ):
        all_results = {}
        for asset_currency in self.asset_currency_list:
            backtest_task_results = []
            self.kwargs['asset_currency'] = asset_currency
            file_path = f"backtest_results/{asset_currency}/{asset_currency}_optimized_result.csv"
            if os.path.exists(file_path):
                result_df = pd.read_csv(file_path, index_col=0)
                result_df = result_df[result_df['sharpe'] > self.kwargs['minimum_sharpe']].sort_values(by='strategy').reset_index(drop=True)
                backtest_task_results = self.prepare_backtest_tasks(result_df, self.kwargs)
                # result_list.append(backtest_task_results)
            all_results.update({asset_currency: backtest_task_results})
        return all_results

    @staticmethod
    def prepare_backtest_tasks(result_df, kwargs):
        dataframe_key_list = result_df['strategy'].str.rsplit('|', n=5).str[:-5].str.join('|').unique().tolist()
        backtest_dataframe_map = {strategy_key: CryptoDataService({
            **kwargs,
            'asset_currency': key_parts[1],
            'data_source': key_parts[3],
            'factor_currency': key_parts[0],
            'timeframe': key_parts[2],
            'endpoint': key_parts[-1]
        }).create_backtest_dataframe(CryptoExchangeDataService(**{**kwargs, 'timeframe': key_parts[2]}).get_historical_data(True)) for strategy_key in dataframe_key_list for key_parts in [strategy_key.split('|')]}

        strategy_keys = result_df['strategy'].str.rsplit('|', n=5).str[:].str.join('|').unique().tolist()
        timeframe_list = list({strategy_key.split('|')[2] for strategy_key in dataframe_key_list})

        all_backtest_results = []
        for timeframe in timeframe_list:
            tasks = []
            for strategy_key_full in strategy_keys:
                backtest_dataframe_key = '|'.join(strategy_key_full.rsplit('|', 5)[:-5])
                backtest_df = backtest_dataframe_map[backtest_dataframe_key].copy()
                if strategy_key_full.split('|')[2] == timeframe:
                    strategy_key_parts = strategy_key_full.split('|')
                    tasks.append({
                        'action': strategy_key_parts[-3],
                        'asset_currency': strategy_key_parts[1],
                        'backtest_df': backtest_df,
                        'data_source': strategy_key_parts[3],
                        'endpoint': strategy_key_parts[-6],
                        'factor_currency': strategy_key_parts[0],
                        'generate_equity_curve': kwargs['generate_equity_curve'],
                        'indicator': strategy_key_parts[-5],
                        'lookback_list': [int(strategy_key_parts[-2])],
                        'orientation': strategy_key_parts[-4],
                        'threshold_list': [float(strategy_key_parts[-1])],
                        'timeframe': strategy_key_parts[2],
                    })
            if tasks:
                backtest_task_results = Utilities.run_in_parallel(EquityCurveGenerator.evaluate_backtest, tasks) if len(tasks) > 1 else [EquityCurveGenerator.evaluate_backtest(tasks[0])]
                all_backtest_results.append(backtest_task_results)
        return all_backtest_results

    @staticmethod
    def evaluate_backtest(task, **kwargs):
        return BacktestEngine.performance_evaluation(task)


    @staticmethod
    def process_results(all_results, asset_currency_list, show_equity_curve):
        for asset_currency in asset_currency_list:
            backtest_results_folder = f"backtest_results/{asset_currency}"
            os.makedirs(backtest_results_folder, exist_ok=True)
            different_timeframe_results = all_results[asset_currency]
            tasks = []
            for results in different_timeframe_results:
                if len(results) > 1: show_equity_curve = False
                for equity_curve_data in results:
                    tasks.append({
                        'backtest_results_folder': backtest_results_folder,
                        'equity_curve_data': equity_curve_data,
                        'show_equity_curve': show_equity_curve,
                    })


                Utilities.run_in_parallel(EquityCurveGenerator.save_equity_curve, tasks) if len(tasks) > 1 else EquityCurveGenerator.save_equity_curve(tasks[0])


    @staticmethod
    def save_equity_curve(task):
        backtest_results_folder = task['backtest_results_folder']
        equity_curve_data = task['equity_curve_data']
        show_equity_curve = task['show_equity_curve']

        df = equity_curve_data['backtest_df']
        calmar = equity_curve_data['calmar']
        max_drawdown_days = equity_curve_data['max_drawdown_days']
        mdd = equity_curve_data['mdd']
        pos_count = equity_curve_data['pos_count']
        sharpe = equity_curve_data['sharpe']
        trades = equity_curve_data['trades']
        x = int(equity_curve_data['x'])
        y = float(equity_curve_data['y'])
        title = f"{equity_curve_data['title'].rsplit('|', 1)[0]}|x{int(equity_curve_data['x'])}|y{y}\nsharpe{sharpe}|mdd{mdd}|calmar{calmar}|max_dd_days{max_drawdown_days}|pos_count{pos_count}"

        if show_equity_curve:
            fig = px.line(df, x=df.index, y=['cumu', 'dd', 'benchmark_cumu'], title=title)
            fig.show()
        else:
            df.plot(figsize=(10, 6))
            plt.title(title)
            result_output_path = os.path.join(backtest_results_folder, f"{equity_curve_data['title'].split('|')[7]}/{equity_curve_data['title'].split('|')[2]}/{equity_curve_data['title'].rsplit('|', 1)[0]}")
            os.makedirs(result_output_path, exist_ok=True)
            equity_curve_file_name = f"{equity_curve_data['title'].rsplit('|', 1)[0]}|file2|{x}|{y}|{df.index[0].strftime('%Y-%m-%d')}.png"

            plt.savefig(f"{result_output_path}/{equity_curve_file_name}")
            plt.close()
