import os
import math
import sys

import numpy as np
import pandas as pd
import multiprocessing as mp

import seaborn as sns
import matplotlib.pyplot as plt


class Utilities:

    TIMEFRAME_TIMEUNIT_MAP = {
        '1m': 365 * 24 * 60,
        '2m': 365 * 24 * 30,
        '3m': 365 * 24 * 20,
        '4m': 365 * 24 * 15,
        '5m': 365 * 24 * 12,
        '10m': 365 * 24 * 6,
        '15m': 365 * 24 * 4,
        '30m': 365 * 24 * 2,
        '1h': 365 * 24,
        '2h': 365 * 12,
        '3h': 365 * 8,
        '4h': 365 * 6,
        '6h': 365 * 4,
        '8h': 365 * 3,
        '12h': 365 * 2,
        '24h': 365,
        '1d': 365,
        '1w': 365 // 7,
        '1month': 12}
    TIMEFRAME_DAYS_MAP = {
        '1m': 1 / (24 * 60),  # 1 minute
        '2m': 2 / (24 * 60),  # 2 minutes
        '3m': 3 / (24 * 60),  # 3 minutes
        '4m': 4 / (24 * 60),  # 4 minutes
        '5m': 5 / (24 * 60),  # 5 minutes
        '10m': 10 / (24 * 60),  # 10 minutes
        '15m': 15 / (24 * 60),  # 15 minutes
        '30m': 30 / (24 * 60),  # 30 minutes
        '1h': 1 / 24,  # 1 hour
        '2h': 2 / 24,  # 2 hours
        '3h': 3 / 24,  # 3 hours
        '4h': 4 / 24,  # 4 hours
        '5h': 5 / 24,  # 5 hours
        '6h': 6 / 24,  # 6 hours
        '7h': 7 / 24,  # 7 hours
        '8h': 8 / 24,  # 8 hours
        '9h': 9 / 24,  # 9 hours
        '10h': 10 / 24,  # 10 hours
        '11h': 11 / 24,  # 11 hours
        '12h': 12 / 24,  # 12 hours
        '24h': 1,  # 24 hours (1 day)
        '1d': 1,  # 1 day
        '1w': 7,  # 1 week
        '1week': 7,  # 1 week
        '2w': 14,  # 2 weeks
        '2week': 14,  # 2 weeks
        '1M': 30,  # 1 month
        '1month': 30  # 1 month
    }

    def __init__(self):

        pass

    @classmethod
    def generate_lookback_lists(cls, df, first_step=None, max_lookback=None, lookback_step=None):
        if first_step is None: first_step = 2
        if max_lookback is None:
            df.index = pd.to_datetime(df['unix_timestamp'], unit='ms')
            max_3_months_lookback = (len(df[:df.index[0] + pd.DateOffset(months=3)]) // 100 + 1) * 100
            max_lookback = min(len(df) // 10, max_3_months_lookback)
        if lookback_step is None: lookback_step = 5

        lookback_list = [
            2, 3, 8, 11, 14, 18, 22, 26, 30, 35, 40, 45, 50,
            *[int(x) for x in np.arange(55, 105, 5)],
            *[int(x) for x in np.arange(110, 301, 10)]
        ]

        current_value = lookback_list[-1] + 15
        while current_value <= max_lookback:
            lookback_list.append(current_value)
            current_value = int(current_value * 1.035)
        lookback_list = lookback_list[:100] if len(lookback_list) > 100 else lookback_list
        return lookback_list

    @classmethod
    def generate_threshold_list(cls, df: pd.DataFrame, indicator: str, max_threshold: float, number_of_intervals: int) -> np.ndarray:
        """
        Generates a list of threshold values based on the given indicator and parameters.

        Parameters:
        - df: DataFrame containing the data.
        - indicator: Type of indicator ('bband' or others).
        - max_threshold: The maximum threshold value.
        - number_of_intervals: The number of intervals for threshold calculation.

        Returns:
        - A NumPy array of threshold values.
        """

        def calculate_step_size(max_threshold: float, number_of_intervals: int) -> float:
            """Calculates the step size based on the max threshold and number of intervals."""

            if max_threshold == 0: return 0
            y = max_threshold / number_of_intervals
            if max_threshold > 100: return int(y)
            elif max_threshold > 10: return round(y, 1)
            elif max_threshold > 1: return round(y, 2)
            elif max_threshold >= 0.1: return round(y, 3)
            elif max_threshold >= 0.01: return round(y, 4)
            elif max_threshold >= 0.001: return round(y, 5)
            else: return round(y, 6)

        first_threshold_step = 0
        x = 2
        if not number_of_intervals: number_of_intervals = 10

        if indicator == 'bband':
            if not max_threshold:
                df['ma'] = df['factor'].rolling(x).mean()
                df['sd'] = df['factor'].rolling(x).std()
                df['z'] = (df['factor'] - df['ma']) / df['sd']
                max_threshold = np.nanstd(df['z'].replace([np.inf, -np.inf], np.nan))
                if np.isnan(max_threshold): max_threshold = 2.5
            else: max_threshold = max(0.1, max_threshold)

        elif indicator == 'rsi':
            if not max_threshold:
                df['delta'] = df['factor'].diff(1)
                df['delta'] = df['delta'].astype(float).fillna(0)
                df['positive'] = df['delta'].clip(lower=0)
                df['negative'] = df['delta'].clip(upper=0)
                df['average_gain'] = df['positive'].rolling(x).mean()
                df['average_loss'] = abs(df['negative'].rolling(x).mean())
                df['relative_strength'] = df['average_gain'] / df['average_loss']
                df['z'] = 100 - (100 / (1 + df['relative_strength']))
                max_threshold = np.nanstd(df['z'].replace([np.inf, -np.inf], np.nan))
            else: max_threshold = max(0.1, min(max_threshold, 55.5))

        elif indicator == 'ma_diff':
            if not max_threshold:
                df['ma'] = df['factor'].rolling(x).mean()
                df['z'] = df['factor'] / df['ma'] - 1
                max_threshold = np.nanstd(df['z'].replace([np.inf, -np.inf], np.nan))
            else: max_threshold = max(0.000001, max_threshold)

        elif indicator == 'roc':
            if not max_threshold:
                df['z'] = df['factor'].pct_change(periods=x) * 100
                max_threshold = np.nanstd(df['z'].replace([np.inf, -np.inf], np.nan))
            else: max_threshold = max(0.000001, max_threshold)

        elif indicator == 'ma_roc':
            if not max_threshold:
                df['ma'] = df['factor'].rolling(x).mean()
                df['z'] = df['ma'].pct_change(periods=1) * 100
                max_threshold = np.nanstd(df['z'].replace([np.inf, -np.inf], np.nan))
            else: max_threshold = max(0.000001, max_threshold)

        elif indicator == 'percentile_rank':
            if not max_threshold:
                df['z'] = df['factor'].rank(pct=True) * 100 - 50
                max_threshold = np.nanstd(df['z'].replace([np.inf, -np.inf], np.nan))
                pass

        elif indicator == 'roc_hv':
            if not max_threshold:
                x = max(x, 2)
                df['roc'] = df['factor'].pct_change(periods=x) * 100
                df['std'] = df['price'].rolling(x).std()
                df['roc_z'] = (df['roc'] - df['roc'].mean()) / df['roc'].std()
                df['std_z'] = (df['std'] - df['std'].mean()) / df['std'].std()
                df['z'] = df['roc_z'] + df['std_z']

                max_threshold = np.nanstd(df['z'].replace([np.inf, -np.inf], np.nan))

        elif indicator == 'cross_ma':
            first_threshold_step = 5
            max_threshold = 100
            pass

        max_threshold *= 1.1
        threshold_step = calculate_step_size(max_threshold, number_of_intervals)
        threshold_list = np.round(np.arange(first_threshold_step, max_threshold, threshold_step), 6)
        return threshold_list

    @staticmethod
    def run_in_parallel(func, param_list, max_cores=None):
        if not param_list:
            return []
        num_cores = max_cores or max(1, min(len(param_list), mp.cpu_count()))
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(func, param_list)
        return results

    @classmethod
    def filter_results_by_sharpe_ratio(cls, results_by_currency, min_sharpe_ratio):
        """
        Filter results based on a minimum Sharpe ratio.

        Parameters:
        - results_by_currency (dict): A dictionary containing results categorized by currency pairs.
        - min_sharpe_ratio (float): The minimum Sharpe ratio to filter results.

        Returns:
        - list: A list of unique backtest dataframe keys.
        - list: A list of filtered results.
        - list: A list of result data dictionaries.
        """

        unique_backtest_keys = set()
        filtered_results = []
        result_data_list = []

        for currency_pair, results in results_by_currency.items():
            for result_entry in results:
                for data in result_entry:
                    sharpe_ratio = data['result']['sharpe']
                    if sharpe_ratio >= min_sharpe_ratio:
                        unique_backtest_keys.add(data['backtest_dataframe_key'])
                        filtered_results.append(data)
                        result_data_list.append(data['result'])

        return list(unique_backtest_keys), filtered_results, result_data_list

    @classmethod
    def generate_heatmap(cls, all_results, show_heatmap=False):

        asset_currency_list = list(all_results.keys())
        for asset_currency in asset_currency_list:
            results = list(all_results[asset_currency])
            total_heatmaps = list(results)
            for heatmap in results:
                heatmap_data = []
                for param_combos in heatmap:
                    heatmap_data.append(param_combos['result'])

                title = f"{heatmap[0]['factor_currency']}|{heatmap[0]['asset_currency']}|{heatmap[0]['timeframe']}|{heatmap[0]['data_source']}|{heatmap[0]['endpoint']}|{heatmap[0]['indicator']}|{heatmap[0]['orientation']}|{heatmap[0]['action']}"
                chunks = [heatmap_data[i:i + 250] for i in range(0, len(heatmap_data), 250)]
                if len(chunks) >= 2:
                    if len(chunks) % 2 != 0: chunks.pop()
                    for i in range(0, len(chunks), 2):
                        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10))
                        data_table1 = pd.DataFrame(chunks[i]).pivot_table(index='x', columns='y', values='sharpe')
                        sns.heatmap(data_table1, ax=ax1, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})
                        ax1.yaxis.set_tick_params(rotation=0)

                        if i + 1 < len(chunks):
                            # subfolder_path2 = heatmap_data.get('subfolder_path', '')
                            plt.suptitle(f"{title}\nFrom: {heatmap[0]['since']} to {heatmap[0]['end']}, Data quantity: {heatmap[0]['data_quantity']}")
                            data_table2 = pd.DataFrame(chunks[i+1]).pivot_table(index='x', columns='y', values='sharpe')
                            sns.heatmap(data_table2, ax=ax2, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})
                            ax2.yaxis.set_tick_params(rotation=0)
                        if show_heatmap:
                            plt.show()
                        else:
                            heatmap_file_path = os.path.join(subfolder_path1, f"{title1}_file{i + 1}")
                            fig.savefig(f"{heatmap_file_path}_all_time")
                        plt.close(fig)


                else:
                    plt.suptitle(f"{title}\nFrom: {heatmap[0]['since']} to {heatmap[0]['end']}, Data quantity: {heatmap[0]['data_quantity']}")
                    fig, ax = plt.subplots(figsize=(10, 7))
                    data_table = pd.DataFrame(heatmap_data).pivot_table(index='x', columns='y', values='sharpe')
                    sns.heatmap(data_table, ax=ax, annot=True, fmt='g', cmap='Greens', annot_kws={
                        'fontsize': 8})
                    ax.yaxis.set_tick_params(rotation=0)
                    plt.show()
                    plt.close(fig)

                    if show_heatmap:
                        plt.show()
                    # else:
                    #     heatmap_file_path = os.path.join(subfolder_path, f"{title}_file1")
                    #     fig.savefig(f"{heatmap_file_path}_all_time")
                    plt.close(fig)

                sys.exit()
                # Create the heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', cbar=True)

                # Set titles and labels
                plt.title('Heatmap of Sharpe Ratios')
                plt.xlabel('y')
                plt.ylabel('x')

                # Show the heatmap
                plt.show()

                pass

            if total_heatmaps == 1:
                heatmap_data = results[0]
                title = heatmap_data['title']
                subfolder_path = heatmap_data.get('subfolder_path', '')
                fig, ax = plt.subplots(figsize=(10, 7))
                plt.suptitle(f"{title}\nFrom: {heatmap_data['since']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                result_list = heatmap_data['result']
                data_table = pd.DataFrame(result_list).pivot_table(index='x', columns='y', values='sharpe')
                sns.heatmap(data_table, ax=ax, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})

                ax.yaxis.set_tick_params(rotation=0)

                if show_heatmap:
                    plt.show()
                else:
                    heatmap_file_path = os.path.join(subfolder_path, f"{title}_file1")
                    fig.savefig(f"{heatmap_file_path}_all_time")
                plt.close(fig)

            else:
                if total_heatmaps % 2 == 0:
                    for i in range(0, total_heatmaps, 2):
                        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10))
                        plt.subplots_adjust(left=0.06, bottom=0.12, right=1, top=0.9, wspace=0.15)

                        heatmap_data = total_heatmaps[i]
                        title = f"{heatmap[0]['factor_currency']}|{heatmap[0]['asset_currency']}|{heatmap[0]['timeframe']}|{heatmap[0]['data_source']}|{heatmap[0]['endpoint']}|{heatmap[0]['indicator']}|{heatmap[0]['orientation']}|{heatmap[0]['action']}|{heatmap[0]['result']['x']}|{heatmap[0]['result']['y']}, sharpe {heatmap[0]['result']['sharpe']}, mdd {heatmap[0]['result']['mdd']}, pos_count {heatmap[0]['result']['pos_count']}"
                        # subfolder_path1 = heatmap_data.get('subfolder_path', '')
                        plt.suptitle(f"{title1}\nFrom: {heatmap_data['since']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                        result_list1 = heatmap_data['result']
                        for result in result_list1: result['y'] = float(result['y'])  # Convert to float
                        if any(isinstance(i['y'], float) and 'e' in str(i['y']) for i in result_list1):
                            result_list1 = [{**result, 'y': f"{result['y']:.6f}"} for result in result_list1]

                        data_table1 = pd.DataFrame(result_list1).pivot_table(index='x', columns='y', values='sharpe')
                        sns.heatmap(data_table1, ax=ax1, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})

                        ax1.yaxis.set_tick_params(rotation=0)

                        if i + 1 < total_heatmaps:
                            heatmap_data = total_heatmaps[i + 1]
                            title2 = heatmap_data['title']
                            # subfolder_path2 = heatmap_data.get('subfolder_path', '')
                            plt.suptitle(f"{title2}\nFrom: {heatmap_data['since']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                            result_list2 = heatmap_data['result']
                            for result in result_list2: result['y'] = float(result['y'])
                            if any(isinstance(i['y'], float) and 'e' in str(i['y']) for i in
                                   result_list2): result_list2 = [{**result, 'y': f"{result['y']:.6f}"} for result in result_list2]

                            data_table2 = pd.DataFrame(result_list2).pivot_table(index='x', columns='y', values='sharpe')
                            sns.heatmap(data_table2, ax=ax2, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})

                            ax2.yaxis.set_tick_params(rotation=0)

                        if show_heatmap:
                            plt.show()
                        else:
                            heatmap_file_path = os.path.join(subfolder_path1, f"{title1}_file{i + 1}")
                            fig.savefig(f"{heatmap_file_path}_all_time")
                        plt.close(fig)

                else:
                    for i in range(0, total_heatmaps, 2):
                        if i + 1 < total_heatmaps:
                            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10))
                            plt.subplots_adjust(left=0.06, bottom=0.12, right=1, top=0.9, wspace=0.15)

                            heatmap_data = total_heatmaps[i]
                            title1 = heatmap_data['title']
                            subfolder_path1 = heatmap_data.get('subfolder_path', '')
                            plt.suptitle(f"{title1}\nFrom: {heatmap_data['start']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                            result_list1 = heatmap_data['result_list']
                            for result in result_list1: result['y'] = float(result['y'])  # Convert to float
                            if any(isinstance(i['y'], float) and 'e' in str(i['y']) for i in
                                   result_list1): result_list1 = [{**result,'y': f"{result['y']:.6f}"} for result in result_list1]

                            data_table1 = pd.DataFrame(result_list1).pivot_table(index='x', columns='y', values='sharpe')
                            sns.heatmap(data_table1, ax=ax1, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})

                            heatmap_data = total_heatmaps[i + 1]
                            title2 = heatmap_data['title']
                            plt.suptitle(f"{title2}\nFrom: {heatmap_data['start']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                            result_list2 = heatmap_data['result_list']
                            for result in result_list2: result['y'] = float(result['y'])  # Convert to float
                            if any(isinstance(i['y'], float) and 'e' in str(i['y']) for i in
                                   result_list2): result_list2 = [{**result, 'y': f"{result['y']:.6f}"} for result in result_list2]
                            data_table2 = pd.DataFrame(result_list2).pivot_table(index='x', columns='y', values='sharpe')
                            sns.heatmap(data_table2, ax=ax2, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})

                            if show_heatmap:
                                plt.show()
                            else:
                                heatmap_file_path = os.path.join(subfolder_path1, f"{title1}_file{i + 1}")
                                fig.savefig(f"{heatmap_file_path}_all_time")
                            plt.close(fig)

                        else:
                            heatmap_data = total_heatmaps[i]
                            title = heatmap_data['title']
                            subfolder_path = heatmap_data.get('subfolder_path', '')
                            fig, ax = plt.subplots(figsize=(10, 7))
                            plt.suptitle(f"{title}\nFrom: {heatmap_data['start']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                            result_list3 = heatmap_data['result_list']
                            data_table = pd.DataFrame(result_list3).pivot_table(index='x', columns='y', values='sharpe')
                            sns.heatmap(data_table, ax=ax, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})
                            if show_heatmap:
                                plt.show()
                            else:
                                heatmap_file_path = os.path.join(subfolder_path, f"{title}_file{i + 1}")
                                fig.savefig(f"{heatmap_file_path}_all_time")
                            plt.close(fig)




            for heatmap in total_heatmaps:
                title = f"{heatmap[0]['factor_currency']}|{heatmap[0]['asset_currency']}|{heatmap[0]['timeframe']}|{heatmap[0]['data_source']}|{heatmap[0]['endpoint']}|{heatmap[0]['indicator']}|{heatmap[0]['orientation']}|{heatmap[0]['action']}|{heatmap[0]['result']['x']}|{heatmap[0]['result']['y']}, sharpe {heatmap[0]['result']['sharpe']}, mdd {heatmap[0]['result']['mdd']}, pos_count {heatmap[0]['result']['pos_count']}"

                for result in heatmap:
                    for data in result:
                        title = f"{data['factor_currency']}|{data['asset_currency']}|{data['timeframe']}|{data['data_source']}|{data['endpoint']}|{data['indicator']}|{data['orientation']}|{data['action']}\nx {data['result']['x']}, y {data['result']['y']}, sharpe {data['result']['sharpe']}, mdd {data['result']['mdd']}, pos_count {data['result']['pos_count']}"
                        pass

        for group_currency in all_results:
            for heatmap_dict in group_currency:
                total_heatmaps = len(heatmap_dict)

                if total_heatmaps == 1:
                    heatmap_data = heatmap_dict[0]
                    title = heatmap_data['title']
                    subfolder_path = heatmap_data.get('subfolder_path', '')
                    fig, ax = plt.subplots(figsize=(10, 7))
                    plt.suptitle(f"{title}\nFrom: {heatmap_data['since']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                    result_list = heatmap_data['result']
                    data_table = pd.DataFrame(result_list).pivot_table(index='x', columns='y', values='sharpe')
                    sns.heatmap(data_table, ax=ax, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})

                    ax.yaxis.set_tick_params(rotation=0)

                    if show_heatmap:
                        plt.show()
                    else:
                        heatmap_file_path = os.path.join(subfolder_path, f"{title}_file1")
                        fig.savefig(f"{heatmap_file_path}_all_time")
                    plt.close(fig)

                else:
                    if total_heatmaps % 2 == 0:
                        for i in range(0, total_heatmaps, 2):
                            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10))
                            plt.subplots_adjust(left=0.06, bottom=0.12, right=1, top=0.9, wspace=0.15)

                            heatmap_data = heatmap_dict[i]
                            title1 = heatmap_data['title']
                            # subfolder_path1 = heatmap_data.get('subfolder_path', '')
                            plt.suptitle(f"{title1}\nFrom: {heatmap_data['since']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                            result_list1 = heatmap_data['result']
                            for result in result_list1: result['y'] = float(result['y'])  # Convert to float
                            if any(isinstance(i['y'], float) and 'e' in str(i['y']) for i in result_list1): result_list1 = [{**result, 'y': f"{result['y']:.6f}"} for result in result_list1]

                            data_table1 = pd.DataFrame(result_list1).pivot_table(index='x', columns='y', values='sharpe')
                            sns.heatmap(data_table1, ax=ax1, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})

                            ax1.yaxis.set_tick_params(rotation=0)

                            if i + 1 < total_heatmaps:
                                heatmap_data = heatmap_dict[i + 1]
                                title2 = heatmap_data['title']
                                # subfolder_path2 = heatmap_data.get('subfolder_path', '')
                                plt.suptitle(f"{title2}\nFrom: {heatmap_data['since']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                                result_list2 = heatmap_data['result']
                                for result in result_list2: result['y'] = float(result['y'])
                                if any(isinstance(i['y'], float) and 'e' in str(i['y']) for i in result_list2): result_list2 = [{**result, 'y': f"{result['y']:.6f}"} for result in result_list2]

                                data_table2 = pd.DataFrame(result_list2).pivot_table(index='x', columns='y', values='sharpe')
                                sns.heatmap(data_table2, ax=ax2, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})

                                ax2.yaxis.set_tick_params(rotation=0)

                            if show_heatmap:
                                plt.show()
                            else:
                                heatmap_file_path = os.path.join(subfolder_path1, f"{title1}_file{i + 1}")
                                fig.savefig(f"{heatmap_file_path}_all_time")
                            plt.close(fig)

                    else:
                        for i in range(0, total_heatmaps, 2):
                            if i + 1 < total_heatmaps:
                                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10))
                                plt.subplots_adjust(left=0.06, bottom=0.12, right=1, top=0.9, wspace=0.15)

                                heatmap_data = heatmap_dict[i]
                                title1 = heatmap_data['title']
                                subfolder_path1 = heatmap_data.get('subfolder_path', '')
                                plt.suptitle(f"{title1}\nFrom: {heatmap_data['start']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                                result_list1 = heatmap_data['result_list']
                                for result in result_list1: result['y'] = float(result['y'])  # Convert to float
                                if any(isinstance(i['y'], float) and 'e' in str(i['y']) for i in result_list1): result_list1 = [{**result, 'y': f"{result['y']:.6f}"} for result in result_list1]

                                data_table1 = pd.DataFrame(result_list1).pivot_table(index='x', columns='y', values='sharpe')
                                sns.heatmap(data_table1, ax=ax1, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})

                                heatmap_data = heatmap_dict[i + 1]
                                title2 = heatmap_data['title']
                                plt.suptitle(f"{title2}\nFrom: {heatmap_data['start']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                                result_list2 = heatmap_data['result_list']
                                for result in result_list2: result['y'] = float(result['y'])  # Convert to float
                                if any(isinstance(i['y'], float) and 'e' in str(i['y']) for i in result_list2): result_list2 = [{**result, 'y': f"{result['y']:.6f}"} for result in result_list2]
                                data_table2 = pd.DataFrame(result_list2).pivot_table(index='x', columns='y', values='sharpe')
                                sns.heatmap(data_table2, ax=ax2, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})

                                if show_heatmap:
                                    plt.show()
                                else:
                                    heatmap_file_path = os.path.join(subfolder_path1, f"{title1}_file{i + 1}")
                                    fig.savefig(f"{heatmap_file_path}_all_time")
                                plt.close(fig)

                            else:
                                heatmap_data = heatmap_dict[i]
                                title = heatmap_data['title']
                                subfolder_path = heatmap_data.get('subfolder_path', '')
                                fig, ax = plt.subplots(figsize=(10, 7))
                                plt.suptitle(f"{title}\nFrom: {heatmap_data['start']} to {heatmap_data['end']}, Data quantity: {heatmap_data['data_quantity']}")
                                result_list3 = heatmap_data['result_list']
                                data_table = pd.DataFrame(result_list3).pivot_table(index='x', columns='y', values='sharpe')
                                sns.heatmap(data_table, ax=ax, annot=True, fmt='g', cmap='Greens', annot_kws={'fontsize': 8})
                                if show_heatmap:
                                    plt.show()
                                else:
                                    heatmap_file_path = os.path.join(subfolder_path, f"{title}_file{i + 1}")
                                    fig.savefig(f"{heatmap_file_path}_all_time")
                                plt.close(fig)



# import plotly.express as px
# fig = px.line(df, x=df.index, y=['z'], title='')
# fig.show()