import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error



from crypto_exchange_data_service import CryptoExchangeDataService

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


def strategy_effectiveness(action, all_lookback_lists, df, indicator, orientation, threshold_list, timeframe, title, **kwargs):

    save_result = False
    all_results = []
    cv_results = []

    for lookback_list in all_lookback_lists:
        results = []
        for x in lookback_list:
            for y in threshold_list:
                backtest_df = df.copy()
                parameters = {
                    'df': backtest_df,
                    'indicator': indicator,
                    'orientation': orientation,
                    'action': action,
                    'timeframe': timeframe,
                    'x': x,
                    'y': y, }
                result = performance_evaluation(parameters, x, y)
                results.append(result)


                if result['sharpe'] >= 1:
                    save_result = True
                if cross_validate:
                    combined_result = {
                        'x': x,
                        'y': y, **result  # unpack sharpe, cumu, dd, etc.
                    }
                    cv_scores = cross_validation(backtest_df, indicator, x)
                    for model_name, score in cv_scores.items():
                        combined_result[f'{model_name}_mean_mse'] = score['mean_mse']
                    cv_results.append(combined_result)
        all_results.append({
            'since': df.index[0],
            'end': df.index[-1],
            # 'title': title,
            'result': results,
            'data_quantity': len(df), })
    if cv_results:
        cv_df = pd.DataFrame(cv_results).sort_values('sharpe', ascending=False).reset_index(drop=True)
        print(cv_df.head())
    if save_result is True:
        return all_results

def performance_evaluation(parameters, x, y):
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
                'reversion_short_only': lambda z: np.where(z > y, -1, 0), }
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
                'reversion_short_only': lambda z: np.where(z > upper_bond, -1, 0), }
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
                'momentum_long_short': np.where(df['fast_ma'] > df['slow_ma'], 1, np.where(df['fast_ma'] < df['slow_ma'], -1, 0)),
                'momentum_long_only': np.where(df['fast_ma'] > df['slow_ma'], 1, 0),
                'momentum_short_only': np.where(df['fast_ma'] < df['slow_ma'], -1, 0),
                'reversion_long_short': np.where(df['fast_ma'] > df['slow_ma'], -1, np.where(df['fast_ma'] < df['slow_ma'], 1, 0)),
                'reversion_long_only': np.where(df['fast_ma'] > df['slow_ma'], -1, 0),
                'reversion_short_only': np.where(df['fast_ma'] < df['slow_ma'], 1, 0), }
            df['pos'] = cross_ma_action_dict[strategy]

        return df

    df = compute_position(**parameters)
    df['pos_count'] = (df['pos'] != 0).cumsum()
    df['trade'] = (df['pos'].diff().abs() > 0).astype(int)
    df['pnl'] = df['pos'].shift(1) * df['chg'] - df['trade'] * 0.06 / 100
    df['cumu'] = df['pnl'].cumsum()
    df['cummax'] = df['cumu'].cummax()
    df['dd'] = df['cummax'] - df['cumu']
    df['benchmark'] = df['chg']
    df.iloc[0:x - 1, df.columns.get_loc('benchmark')] = 0
    df['benchmark_cumu'] = df['benchmark'].cumsum()

    timeunit = TIMEFRAME_TIMEUNIT_MAP[parameters['timeframe']]
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

    return {
        'x': x,
        'y': y,
        'sharpe': sharpe,
        'mdd': mdd,
        'calmar': calmar,
        'benchmark_sharpe': benchmark_sharpe,
        'total_return': cumu,
        'volatility': return_sd
    }

def cross_validation(df, indicator, x):
    def prepare_features(df, indicator, x):
        if indicator == 'bband':
            df['rolling_mean'] = df['factor'].rolling(x).mean()
            df['rolling_std'] = df['factor'].rolling(x).std()
            df['upper_band'] = df['rolling_mean'] + 2 * df['rolling_std']
            df['lower_band'] = df['rolling_mean'] - 2 * df['rolling_std']
            features = df[['factor', 'price', 'upper_band', 'lower_band', 'z']].iloc[x:]
        elif indicator == 'rsi':
            delta = df['price'].diff()
            gain = delta.where(delta> 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(x).mean()
            avg_loss = loss.rolling(x).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            features = df[['price', 'rsi', 'z']].iloc[x:]
        else:
            raise ValueError("Indicator not supported.")
        return features, df['cumu'].iloc[x:]

    def perform_time_series_cv(model, features, target, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, test_idx in tscv.split(features):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            scores.append(mse)
        return np.mean(scores)

    features, target = prepare_features(df, indicator, x)
    models = {
        # 'LinearRegression': LinearRegression(),
        # 'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=2, random_state=42)}
    results = {}
    for name, model in models.items():
        mean_mse = perform_time_series_cv(model, features, target)
        results[name] = {'mean_mse': mean_mse}

    return results



if __name__ == '__main__':

    action = 'long_short'
    asset_currency = 'BTC'
    indicator = 'bband'
    orientation = 'momentum'
    timeframe = '1d'
    strategy = f"{orientation}_{action}"

    cross_validate = True

    lookback_list = list(np.arange(5, 100, 10))
    threshold_list = list(np.arange(0, 2.5, 0.25))

    kwargs = {
        'exchange_name': 'bybit',
        'product_type': 'linear',
        'since': '2020-05-11',
        'timeframe': timeframe,
    }
    exchange_data = CryptoExchangeDataService(asset_currency, **kwargs)
    df = exchange_data.get_historical_data(True)

    df = df[['unix_timestamp', 'close']]
    df = df.rename(columns={'close': 'factor'})
    df['price'] = df['factor']

    df['chg'] = df['price'].pct_change()

    backtest_combos = {key: value for key, value in locals().items() if not key.startswith('__') and isinstance(value, (str, int, float, bool))}

    backtest_combos.update({
        'df': df, 'all_lookback_lists': [lookback_list], 'threshold_list': threshold_list, 'title': f"{asset_currency}_{indicator}_{strategy}"
    })

    result = strategy_effectiveness(**backtest_combos)
    result_df = pd.DataFrame(result).sort_values(by='sharpe', ascending=False).reset_index(drop=True)
    print(result_df.head(5))
    sys.exit()

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