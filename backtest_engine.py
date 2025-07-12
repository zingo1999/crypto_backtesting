

import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error




class BacktestEngine:
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
    def compute_position(cls, df, indicator, action, orientation, x, y, **kwargs):
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
                'reversion_short_only': lambda z: np.where(z > y, -1, 0), }
            df['pos'] = bband_action_map[strategy](df['z'])

        elif indicator == 'rsi':
            delta = df['factor'].diff(1)
            delta = delta.fillna(0)
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(x).mean()
            avg_loss = loss.rolling(x).mean()
            rs = avg_gain / avg_loss
            df['z'] = 100 - (100 / (1 + rs))
            upper_bond, lower_bond = 50 + y, 50 - y
            rsi_action_map = {
                'momentum_long_short': lambda z: np.where(z > upper_bond, 1, np.where(z < lower_bond, -1, 0)),
                'momentum_long_only': lambda z: np.where(z > upper_bond, 1, 0),
                'momentum_short_only': lambda z: np.where(z < lower_bond, -1, 0),
                'reversion_long_short': lambda z: np.where(z > upper_bond, -1, np.where(z < lower_bond, 1, 0)),
                'reversion_long_only': lambda z: np.where(z < lower_bond, 1, 0),
                'reversion_short_only': lambda z: np.where(z > upper_bond, -1, 0), }
            df['pos'] = rsi_action_map[strategy](df['z'])

        elif indicator == 'percentile_rank':

            df['z'] = df['factor'].rank(pct=True)
            df['pos'] = np.where(df['z'] > y, 1, np.where(df['z'] < y, -1, 0))
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
                'reversion_short_only': lambda z: np.where(z > y, -1, 0), }
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
                'reversion_short_only': lambda z: np.where(z > y, -1, 0), }
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
                'reversion_short_only': lambda z: np.where(z > y, -1, 0), }
            df['pos'] = ma_roc_action_map[strategy](df['z'])

            pass

        elif indicator == 'cross_ma':

            df['fast_ma'] = df['factor'].rolling(x).mean()
            df['slow_ma'] = df['factor'].rolling(int(y)).mean()

            cross_ma_action_dict = {
                'momentum_long_short': np.where(df['fast_ma'] > df['slow_ma'],
                                                1,
                                                np.where(df['fast_ma'] < df['slow_ma'], -1, 0)),
                'momentum_long_only': np.where(df['fast_ma'] > df['slow_ma'], 1, 0),
                'momentum_short_only': np.where(df['fast_ma'] < df['slow_ma'], -1, 0),
                'reversion_long_short': np.where(df['fast_ma'] > df['slow_ma'],
                                                 -1,
                                                 np.where(df['fast_ma'] < df['slow_ma'], 1, 0)),
                'reversion_long_only': np.where(df['fast_ma'] > df['slow_ma'], -1, 0),
                'reversion_short_only': np.where(df['fast_ma'] < df['slow_ma'], 1, 0), }
            df['pos'] = cross_ma_action_dict[strategy]

        return df

    @classmethod
    def performance_evaluation(cls, backtest_combos, **kwargs):
        def strategy_effectiveness(action, asset_currency, backtest_df, data_source, endpoint, factor_currency, indicator, lookback_list, minimum_sharpe, orientation, threshold_list, timeframe, **kwargs):
            save_result = False
            backtest_dataframe_key = f"{factor_currency}|{asset_currency}|{timeframe}|{endpoint}"
            result_list = []
            if endpoint.startswith('/'): endpoint = endpoint.lstrip('/')
            endpoint = endpoint.replace('/', '|')
            for x in lookback_list:
                for y in threshold_list:
                    parameters = {
                        'df': backtest_df.copy(),
                        'indicator': indicator,
                        'orientation': orientation,
                        'action': action,
                        'timeframe': timeframe,
                        'x': x,
                        'y': y,
                    }
                    df = cls.compute_position(**parameters)
                    df['pos_count'] = (df['pos'] != 0).cumsum()
                    df['trade'] = (df['pos'].diff().abs() > 0).astype(int)
                    df['pnl'] = df['pos'].shift(1) * df['chg'] - df['trade'] * 0.06 / 100
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

                    timeunit = cls.TIMEFRAME_TIMEUNIT_MAP[parameters['timeframe']]
                    cumu = round(df['cumu'].iloc[-1], 3)
                    mdd = round(df['dd'].max(), 2)

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

                    result = {
                        'x': x,
                        'y': y,
                        'sharpe': sharpe,
                        'mdd': mdd,
                        'calmar': calmar,
                        'max_drawdown_days': max_drawdown_days,
                        'cumu': cumu,
                        # 'annual_return': annual_return,
                        'benchmark_sharpe': benchmark_sharpe,
                        'pos_count': pos_count,
                        'trade_count': trades,
                        'win_rate': win_rate,
                        'strategy': f"{factor_currency}|{asset_currency}|{timeframe}|{indicator}|{orientation}|{action}|{x}|{y}|{data_source}|{endpoint}",
                    }
                    result_list.append({
                            'factor_currency': factor_currency,
                            'asset_currency': asset_currency,
                            'timeframe': timeframe,
                            'endpoint': endpoint,
                            'indicator': indicator,
                            'orientation': orientation,
                            'action': action,
                            'data_source': data_source,
                            'backtest_dataframe_key': backtest_dataframe_key,
                            'result': result,
                            'data_quantity': len(df),
                            'end': df.index[-1],
                            'since': df.index[0],
                    })

                    if sharpe >= minimum_sharpe:
                        save_result = True

            if save_result is True:
                return result_list

        result = strategy_effectiveness(**backtest_combos)
        if result:
            return result







    # import plotly.express as px
    #
    # x = 85
    # y = 0.75
    # parameters = {
    #     'df': backtest_df,
    #     'indicator': indicator,
    #     'orientation': orientation,
    #     'action': action,
    #     'timeframe': timeframe,
    #     'x': x,
    #     'y': y, }
    # plot_result = performance_evaluation(parameters, x, y)
    # fig = px.line(backtest_df, x=backtest_df.index, y=['cumu', 'dd', 'benchmark_cumu'], title=f"x {x} y {y} sharpe {plot_result['sharpe']}")
    # # fig = px.line(df, x=df.index, y=['z'], title='')
    # fig.show()