import os
import math
import sys

import numpy as np
import pandas as pd

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
        '1w': 365 / 7,
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

        all_lookback_list = []
        current_lookback_list = []
        list_count = 1
        last_lookback = 2

        ###

        current_lookback_list = []
        j = 0
        while True:
            value = math.ceil(last_lookback * (1.08 ** j))

            if current_lookback_list and value <= current_lookback_list[-1]: value = current_lookback_list[-1] + 3

            if value > max_lookback and len(current_lookback_list) == 25:
                all_lookback_list.append(current_lookback_list)
                break
            elif len(current_lookback_list) == 25:
                all_lookback_list.append(current_lookback_list)
                # current_lookback_list = [current_lookback_list[-1], value]
                current_lookback_list = [value]
            else:
                current_lookback_list.append(value)
            j += 1

        ###

        # for i in range(first_step, max_lookback + 1, lookback_step):
        #     if list_count < 3 and len(current_lookback_list) < 50:
        #         current_lookback_list.append(i)
        #     elif list_count < 3 and len(current_lookback_list) == 50:
        #         all_lookback_list.append(current_lookback_list)
        #         last_lookback = current_lookback_list[-1] + 5
        #         list_count += 1
        #         current_lookback_list = [i]
        #     else:
        #         current_lookback_list = []
        #         j = 0
        #         while True:
        #             value = math.ceil(last_lookback * (1.035 ** j))
        #             if value > max_lookback and len(current_lookback_list) == 25:
        #                 all_lookback_list.append(current_lookback_list)
        #                 break
        #             elif len(current_lookback_list) == 25:
        #                 all_lookback_list.append(current_lookback_list)
        #                 current_lookback_list = [current_lookback_list[-1], value]
        #             else: current_lookback_list.append(value)
        #             j += 1
        #         break
        if len(all_lookback_list) > 1 and len(all_lookback_list) % 2 == 1:
            return all_lookback_list[:-1]
        return all_lookback_list


    @classmethod
    def generate_threshold_list(cls, df, indicator, max_threshold, number_of_interval, multiple=1.2):
        first_threshold_step = 0
        x = 2
        if not number_of_interval: number_of_interval = 10

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


        elif indicator == 'cross_ma':
            first_threshold_step = 5
            max_threshold = 100

        else: max_threshold = 10

        threshold_step = max_threshold / number_of_interval
        threshold_list = np.round(np.arange(first_threshold_step, max_threshold, threshold_step), 6)
        return threshold_list

    @classmethod
    def alpha_engine(cls, backtest_combos):
        def strategy_effectiveness(action, all_lookback_lists, df, indicator, orientation, threshold_list, timeframe, title, **kwargs):
            def compute_position(df, indicator, action, orientation, x, y, **kwargs):
                strategy = f"{orientation}_{action}"

                if indicator == 'bband':
                    df['ma'] = df['factor'].rolling(x).mean()
                    df['sd'] = df['factor'].rolling(x).std()
                    df['z'] = (df['factor'] - df['ma']) / df['sd']

                    bband_action_map = {
                        'momentum_long_short': lambda z: np.where(z > y, 1, np.where(z < -y, -1, 0)),
                        'momentum_long_only': lambda z: np.where(z > y, 1, 0),
                        'momentum_short_only': lambda z: np.where(z < -y, -1, 0),
                        'reversion_long_short': lambda z: np.where(z > y, -1, np.where(z < -y, 1, 0)),
                        'reversion_long_only': lambda z: np.where(z < -y, 1, 0),
                        'reversion_short_only': lambda z: np.where(z > y, -1, 0),
                    }
                    df['pos'] = bband_action_map[strategy](df['z'])

                elif indicator == 'rsi':
                    df['delta'] = df['factor'].diff(1)
                    df['delta'] = df['delta'].astype(float).fillna(0)
                    df['positive'] = df['delta'].clip(lower=0)
                    df['negative'] = df['delta'].clip(upper=0)
                    df['average_gain'] = df['positive'].rolling(x).mean()
                    df['average_loss'] = abs(df['negative'].rolling(x).mean())
                    df['relative_strength'] = df['average_gain'] / df['average_loss']
                    df['z'] = 100 - (100 / (1 + df['relative_strength']))
                    upper_bond, lower_bond = 50 + y, 50 - y
                    rsi_action_map = {
                        'momentum_long_short': lambda z: np.where(z > upper_bond, 1, np.where(z < lower_bond, -1, 0)),
                        'momentum_long_only': lambda z: np.where(z > upper_bond, 1, 0),
                        'momentum_short_only': lambda z: np.where(z < lower_bond, -1, 0),
                        'reversion_long_short': lambda z: np.where(z > upper_bond, -1, np.where(z < lower_bond, 1, 0)),
                        'reversion_long_only': lambda z: np.where(z < lower_bond, 1, 0),
                        'reversion_short_only': lambda z: np.where(z > upper_bond, -1, 0),
                    }
                    df['pos'] = rsi_action_map[strategy](df['z'])

                elif indicator == 'percentile_rank':

                    df['z'] = df['factor'].rank(pct=True)
                    df['pos'] = np.where(df['z'] > y, 1, np.where(df['z'] < (100 - y), -1, 0))
                    pass

                elif indicator == 'ma_diff':
                    df['ma'] = df['factor'].rolling(x).mean()
                    df['z'] = df['factor'] / df['ma'] - 1

                    ma_diff_action_map = {
                        'momentum_long_short': lambda z: np.where(z > y, 1, np.where(z < -y, -1, 0)),
                        'momentum_long_only': lambda z: np.where(z > y, 1, 0),
                        'momentum_short_only': lambda z: np.where(z < -y, -1, 0),
                        'reversion_long_short': lambda z: np.where(z > y, -1, np.where(z < -y, 1, 0)),
                        'reversion_long_only': lambda z: np.where(z < -y, 1, 0),
                        'reversion_short_only': lambda z: np.where(z > y, -1, 0),
                    }
                    df['pos'] = ma_diff_action_map[strategy](df['z'])

                elif indicator == 'ma':
                    df['ma'] = df['factor'].rolling(x).mean()
                    df['z'] = (df['factor'] - df['ma']) / df['factor']

                elif indicator == 'roc':

                    df['z'] = df['factor'].pct_change(periods=x) * 100

                    roc_action_map = {
                        'momentum_long_short': lambda z: np.where(z > y, 1, np.where(z < -y, -1, 0)),
                        'momentum_long_only': lambda z: np.where(z > y, 1, 0),
                        'momentum_short_only': lambda z: np.where(z < -y, -1, 0),
                        'reversion_long_short': lambda z: np.where(z > y, -1, np.where(z < -y, 1, 0)),
                        'reversion_long_only': lambda z: np.where(z < -y, 1, 0),
                        'reversion_short_only': lambda z: np.where(z > y, -1, 0),
                    }
                    df['pos'] = roc_action_map[strategy](df['z'])

                elif indicator == 'ma_roc':

                    df['ma'] = df['factor'].rolling(x).mean()
                    df['z'] = df['ma'].pct_change(periods=1) * 100

                    ma_roc_action_map = {
                        'momentum_long_short': lambda z: np.where(z > y, 1, np.where(z < -y, -1, 0)),
                        'momentum_long_only': lambda z: np.where(z > y, 1, 0),
                        'momentum_short_only': lambda z: np.where(z < -y, -1, 0),
                        'reversion_long_short': lambda z: np.where(z > y, -1, np.where(z < -y, 1, 0)),
                        'reversion_long_only': lambda z: np.where(z < -y, 1, 0),
                        'reversion_short_only': lambda z: np.where(z > y, -1, 0),
                    }
                    df['pos'] = ma_roc_action_map[strategy](df['z'])


                    pass

                elif indicator == 'cross_ma':

                    df['fast_ma'] = df['factor'].rolling(x).mean()
                    df['slow_ma'] = df['factor'].rolling(int(y)).mean()

                    cross_ma_action_dict = {
                        'momentum_long_short': np.where(df['fast_ma'] > df['slow_ma'], 1, np.where(df['fast_ma'] < df['slow_ma'], -1, 0)),
                        'momentum_long_only': np.where(df['fast_ma'] > df['slow_ma'], 1, 0),
                        'momentum_short_only': np.where(df['fast_ma'] < df['slow_ma'], -1, 0),
                        'reversion_long_short': np.where(df['fast_ma'] > df['slow_ma'], -1, np.where(df['fast_ma'] < df['slow_ma'], 1, 0)),
                        'reversion_long_only': np.where(df['fast_ma'] > df['slow_ma'], -1, 0),
                        'reversion_short_only': np.where(df['fast_ma'] < df['slow_ma'], 1, 0),
                    }
                    df['pos'] = cross_ma_action_dict[strategy]

                return df['pos']

            save_result = False
            all_results = []
            for lookback_list in all_lookback_lists:
                result_list = []
                for x in lookback_list:
                    for y in threshold_list:
                        parameters = {
                            'df': df.copy(),
                            'indicator': indicator,
                            'orientation': orientation,
                            'action': action,
                            'timeframe': timeframe,
                            'x': x,
                            'y': y,
                        }
                        # df = df.iloc[:, :3]
                        df['chg'] = df['price'].pct_change()
                        df['pos'] = compute_position(**parameters)
                        df['pos_count'] = (df['pos'] != 0).cumsum()
                        df['trade'] = (df['pos'].diff().abs() > 0).astype(int)

                        df['pnl'] = df['pos'].shift(1) * df['chg'] - df['trade'] * 0.06 / 100
                        # df['pnl'] = df['pos'].shift(1) * df['chg']

                        df['cumu'] = df['pnl'].cumsum()
                        df['cummax'] = df['cumu'].cummax()
                        df['dd'] = df['cummax'] - df['cumu']
                        df['benchmark'] = df['chg']
                        df.iloc[0:x - 1, df.columns.get_loc('benchmark')] = 0
                        df['benchmark_cumu'] = df['benchmark'].cumsum()

                        df['trade_outcome'] = np.where(df['pnl'] > 0, 1, np.where(df['pnl'] < 0, -1, 0))

                        pos_count = round(df['pos_count'].iloc[-1] / len(df), 3)
                        trades = (df['pos'].diff().abs() > 0).sum()

                        drawdown_periods = np.cumsum(np.where(df['cummax'].ne(df['cummax'].shift()), 1, 0))
                        max_drawdown_duration = (df.groupby(drawdown_periods)['cummax'].transform('count') - 1).max()
                        max_drawdown_days = round(max_drawdown_duration * cls.TIMEFRAME_DAYS_MAP[timeframe], 1)

                        total_wins = (df['trade_outcome'] == 1).sum()
                        total_losses = (df['trade_outcome'] == -1).sum()
                        win_rate = (total_wins / (total_wins + total_losses)) if (total_wins + total_losses) > 0 else 0

                        timeunit = cls.TIMEFRAME_TIMEUNIT_MAP[timeframe]
                        cumu = round(df['cumu'].iloc[-1], 3)
                        mdd = round(df['dd'].max(), 2)
                        # annual_return = round(df['pnl'].mean() * timeunit, 3)
                        # calmar = round(annual_return / mdd, 2) if mdd != 0 else 0

                        annual_return = round(df['pnl'].iloc[x - 1:].mean() * timeunit, 3)
                        avg_return = df['pnl'].iloc[x - 1:].mean()
                        return_sd = df['pnl'].iloc[x - 1:].std()
                        sharpe = round(avg_return / return_sd * np.sqrt(timeunit), 2) if annual_return and return_sd else 0
                        calmar = round(avg_return * timeunit / mdd, 2) if mdd != 0 else 0

                        benchmark_mean = df['benchmark'].iloc[x - 1:].mean()
                        benchmark_std = df['benchmark'].iloc[x - 1:].std()
                        benchmark_sharpe = round(benchmark_mean / benchmark_std * np.sqrt(timeunit), 2) if benchmark_mean and benchmark_std else 0

                        if win_rate >= 0.75: print(f"{parameters['indicator']}_{parameters['orientation']}_{x}_{y} - win rate {round(win_rate * 100, 3)}%")

                        if isinstance(y, float) and 'e' in f"{y}": y = f"{y:.10f}".rstrip('0')

                        # return pd.Series([x, y, sharpe, mdd, calmar, max_drawdown_days, cumu, trades, annual_return, benchmark_sharpe, pos_count, win_rate], index=['x', 'y', 'sharpe', 'mdd', 'calmar', 'max_drawdown_days', 'cumu', 'trades', 'annual_return', 'benchmark_sharpe', 'pos_count', 'win_rate'])
                        result = {
                            'x': x,
                            'y': y,
                            'sharpe': sharpe,
                            'mdd': mdd,
                            'calmar': calmar,
                            'max_drawdown_days': max_drawdown_days,
                            'cumu': cumu,
                            'trades': trades,
                            'annual_return': annual_return,
                            'benchmark_sharpe': benchmark_sharpe,
                            'pos_count': pos_count,
                            'trade_count': trades,
                            'win_rate': win_rate, }
                        result_list.append(result)
                        if sharpe >= 1: save_result = True
                all_results.append({
                    'since': df.index[0],
                    'end': df.index[-1],
                    'title': title,
                    'result': result_list,
                    'data_quantity': len(df),
                })
            if save_result is True:
                return all_results

        result = strategy_effectiveness(**backtest_combos)
        if result:
            return result

    @classmethod
    def generate_heatmap(cls, all_heatmaps, show_heatmap=False):

        for group_currency in all_heatmaps:
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