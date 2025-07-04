import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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

    save_result = False
    all_results = []
    cv_results = []

    for lookback_list in all_lookback_lists:
        results = []
        for x in lookback_list:
            for y in threshold_list:
                parameters = {
                    'df': df.copy(),
                    'indicator': indicator,
                    'orientation': orientation,
                    'action': action,
                    'timeframe': timeframe,
                    'x': x,
                    'y': y, }
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
                result = performance_evaluation(df, timeframe, x, y)
                results.append(result)

                combined_result = {
                    'x': x,
                    'y': y, **result  # unpack sharpe, cumu, dd, etc.
                }
                if result['sharpe'] >= 1:
                    save_result = True
                if cross_validate:
                    cv_scores = cross_validation(df.copy(), indicator, x)
                    for model_name, score in cv_scores.items():
                        combined_result[f'{model_name}_mean_mse'] = score['mean_mse']
                        combined_result[f'{model_name}_std_mse'] = score['std_mse']
                    cv_results.append(combined_result)
        all_results.append({
            'since': df.index[0],
            'end': df.index[-1],
            # 'title': title,
            'result': results,
            'data_quantity': len(df), })
    if cv_results:
        cv_df = pd.DataFrame(cv_results).sort_values('sharpe', ascending=False).reset_index(drop=True)
        print(cv_df.head(3))
    if save_result is True:
        return all_results



def performance_evaluation(df, timeframe,  x, y):

    timeunit = TIMEFRAME_TIMEUNIT_MAP[timeframe]
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


# def cross_validation(df, indicator, rolling_window):
#     def perform_cross_validation(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, n_splits=5):
#         kf = KFold(n_splits=n_splits)
#         scores = []
#
#         for train_index, test_index in kf.split(X):
#             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#             y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#             model.fit(X_train, y_train)
#             score = model.score(X_test, y_test)
#             scores.append(score)
#
#         return np.mean(scores), np.std(scores)
#
#     if indicator == 'bband':
#         df['upper_band'] = df['factor'].rolling(rolling_window).mean() + 2 * df['factor'].rolling(rolling_window).std()
#         df['lower_band'] = df['factor'].rolling(rolling_window).mean() - 2 * df['factor'].rolling(rolling_window).std()
#         features = df[['factor', 'price', 'upper_band', 'lower_band', 'z']].iloc[rolling_window:]
#
#     elif indicator == 'rsi':
#         delta = df['price'].diff()
#         gain = delta.where(delta > 0, 0)
#         loss = -delta.where(delta < 0, 0)
#         avg_gain = gain.rolling(rolling_window).mean()
#         avg_loss = loss.rolling(rolling_window).mean()
#         rs = avg_gain / avg_loss
#         df['rsi'] = 100 - (100 / (1 + rs))
#         features = df[['price', 'rsi', 'z']].iloc[rolling_window:]
#
#     else:
#         print("Indicator not supported.")
#         return
#
#     target = df['cumu'].iloc[rolling_window:]
#     model = LinearRegression()
#     mean_score, std_score = perform_cross_validation(model, features, target, n_splits=5)
#     return mean_score, std_score

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
        return np.mean(scores), np.std(scores)

    features, target = prepare_features(df.copy(), indicator, x)

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    results = {}
    for name, model in models.items():
        mean_mse, std_mse = perform_time_series_cv(model, features, target)
        results[name] = {'mean_mse': mean_mse, 'std_mse': std_mse}

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
    sys.exit()



    for x in lookback_list:
        for y in threshold_list:

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
            df['pos_count'] = (df['pos'] != 0).cumsum()
            df['trade'] = (df['pos'].diff().abs() > 0).astype(int)
            df['pnl'] = df['pos'].shift(1) * df['chg'] - df['trade'] * 0.06 / 100

            df['cumu'] = df['pnl'].cumsum()
            df['cummax'] = df['cumu'].cummax()
            df['dd'] = df['cummax'] - df['cumu']
            df['benchmark'] = df['chg']
            df.iloc[0:x - 1, df.columns.get_loc('benchmark')] = 0
            df['benchmark_cumu'] = df['benchmark'].cumsum()

            results.append(performance_evaluation(df, timeframe, x, y))

            if cross_validation is True:
                cross_validation()

    all_results = pd.DataFrame(results).sort_values(by='sharpe', ascending=False).reset_index(drop=True)
    print(all_results.head(5))

    model = LinearRegression()
    mean_score, std_score = cross_validation(model, X, y, n_splits=5)
    print(f'Mean CV Score: {mean_score}, CV Score Std: {std_score}')
    pass













#
# from crypto_exchange_data_service import CryptoExchangeDataService
#
# class MovingAverageBacktester:
#     def __init__(
#         self,
#         data: pd.DataFrame,
#         strategy: str = 'single',
#         ma_window: int = 5,
#         short_window: int = 5,
#         long_window: int = 20,
#         mode: str = 'momentum',
#         initial_cash: float = 10000
# ):
#         self.data = data.copy()
#         self.strategy = strategy
#         self.ma_window = ma_window
#         self.short_window = short_window
#         self.long_window = long_window
#         self.mode = mode
#         self.initial_cash = initial_cash
#         self.cash = initial_cash
#         self.position = 0
#         self.portfolio_value = []
#
#     def apply_strategy(self):
#         if self.strategy == 'single':
#             self._apply_single_ma_strategy()
#         elif self.strategy == 'crossover':
#             self._apply_crossover_strategy()
#         else:
#             raise ValueError("Unsupported strategy. Use 'single' or 'crossover'.")
#
#     def _apply_single_ma_strategy(self):
#         self.data[f'MA{self.ma_window}'] = self.data['close'].rolling(window=self.ma_window).mean()
#         self.data.dropna(inplace=True)
#
#         for _, row in self.data.iterrows():
#             price = row['close']
#             ma = row[f'MA{self.ma_window}']
#
#             if price> ma and self.cash>= price:
#                 self.position = self.cash // price
#                 self.cash -= self.position * price
#             elif price < ma and self.position> 0:
#                 self.cash += self.position * price
#                 self.position = 0
#
#             total_value = self.cash + self.position * price
#             self.portfolio_value.append(total_value)
#
#     def _apply_crossover_strategy(self):
#         self.data['ShortMA'] = self.data['close'].rolling(window=self.short_window).mean()
#         self.data['LongMA'] = self.data['close'].rolling(window=self.long_window).mean()
#         self.data.dropna(inplace=True)
#
#         prev_signal = None
#
#         for _, row in self.data.iterrows():
#             price = row['close']
#             short_ma = row['ShortMA']
#             long_ma = row['LongMA']
#
#             if self.mode == 'momentum':
#                 if short_ma > long_ma and prev_signal != 'buy':
#                     self.position = self.cash // price
#                     self.cash -= self.position * price
#                     prev_signal = 'buy'
#                 elif short_ma < long_ma and prev_signal != 'sell' and self.position > 0:
#                     self.cash += self.position * price
#                     self.position = 0
#                     prev_signal = 'sell'
#
#             elif self.mode == 'reversion':
#                 if short_ma> long_ma and prev_signal != 'sell' and self.position > 0:
#                     self.cash += self.position * price
#                     self.position = 0
#                     prev_signal = 'sell'
#                 elif short_ma < long_ma and prev_signal != 'buy':
#                     self.position = self.cash // price
#                     self.cash -= self.position * price
#                     prev_signal = 'buy'
#
#             total_value = self.cash + self.position * price
#             self.portfolio_value.append(total_value)
#
#     def results(self):
#         final_value = self.portfolio_value[-1]
#         returns = pd.Series(self.portfolio_value).pct_change().dropna()
#
#         if not returns.empty and returns.std() != 0 and not np.isnan(returns.std()):
#             sharpe = returns.mean() / returns.std() * np.sqrt(252)
#         else:
#             sharpe = 0
#
#         drawdowns = pd.Series(self.portfolio_value) / pd.Series(self.portfolio_value).cummax() - 1
#         max_drawdown = drawdowns.min()
#         calmar = (
#                              final_value - self.initial_cash) / self.initial_cash / abs(max_drawdown) if max_drawdown != 0 else np.nan
#
#         result = {
#             'Strategy': self.strategy,
#             'Mode': self.mode,
#             'Initial Cash': self.initial_cash,
#             'Final Portfolio Value': final_value,
#             'Return (%)': (final_value - self.initial_cash) / self.initial_cash * 100,
#             'Sharpe Ratio': sharpe,
#             'Max Drawdown (%)': max_drawdown * 100,
#             'Calmar Ratio': calmar}
#         if self.strategy == 'single':
#             result['MA Window'] = self.ma_window
#         else:
#             result['Short MA'] = self.short_window
#             result['Long MA'] = self.long_window
#         return result
#
#
# def plot_heatmap(df, value_col, title):
#     df = df.drop_duplicates(subset=['Short MA', 'Long MA'])  # ✅ 移除重複組合
#     pivot = df.pivot(index='Short MA', columns='Long MA', values=value_col)
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(pivot, annot=True, fmt=".2f", cmap='Greens', center=0)
#     plt.title(title)
#     plt.xlabel('Long MA')
#     plt.ylabel('Short MA')
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_strategy_vs_benchmark(df, short_ma, long_ma, mode='momentum', initial_cash=10000):
#
#     bt = MovingAverageBacktester(
#         df,
#         strategy='crossover',
#         short_window=short_ma,
#         long_window=long_ma,
#         mode=mode,
#         initial_cash=initial_cash
# )
#     bt.apply_strategy()
#
#     # 策略資產曲線
#     portfolio = pd.Series(bt.portfolio_value, index=df.index[-len(bt.portfolio_value):])
#     cumu = portfolio
#
#     # 策略 drawdown
#     rolling_max = cumu.cummax()
#     drawdown = (cumu / rolling_max - 1) * 100
#
#     # Buy & Hold 策略
#     price = df['close']
#     buy_and_hold = (price / price.iloc[0]) * initial_cash
#     buy_and_hold = buy_and_hold.loc[cumu.index]  # 對齊時間
#
#     # 畫圖
#     plt.figure(figsize=(14, 6))
#
#     plt.subplot(2, 1, 1)
#     plt.plot(cumu, label=f'Strategy ({mode})', color='blue')
#     plt.plot(buy_and_hold, label='Buy & Hold', color='gray', linestyle='--')
#     plt.title(f'Cumulative Portfolio Value (Short MA={short_ma}, Long MA={long_ma}, Mode={mode})')
#     plt.ylabel('Portfolio Value')
#     plt.legend()
#
#     plt.subplot(2, 1, 2)
#     plt.plot(drawdown, label='Drawdown', color='red')
#     plt.title('Drawdown (%)')
#     plt.ylabel('Drawdown %')
#     plt.xlabel('Date')
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#
#     kwargs = {
#         'exchange_name': 'bybit',
#         'product_type': 'linear',
#         'since': '2020-05-11',
#         'timeframe': '1d',
# }
#     exchange_data = CryptoExchangeDataService('BTC', **kwargs)
#     data = exchange_data.get_historical_data(True)
#
#     # short_ma_list = list(np.arange(2, 12, 1))
#     # long_ma_list = list(np.arange(3, 30, 2))
#
#     freq_minutes = (data.index[1] - data.index[0]).total_seconds() / 60
#     minutes_per_month = 30 * 24 * 60
#     max_by_time = int((minutes_per_month * 3) / freq_minutes)
#     max_by_data = int(len(data) * 0.10)
#     max_long_ma = min(max_by_time, max_by_data)
#
#     ma_combinations = []
#     short_ma_list = np.unique(np.round(np.logspace(np.log10(2), np.log10(max_long_ma // 5), num=20)).astype(int))
#     ratios = [3, 6, 9]
#
#     long_ma_list = []
#     for short_ma in short_ma_list:
#         for r in ratios:
#             long_ma = int(short_ma * r)
#             if long_ma <= max_long_ma:
#                 long_ma_list.append(long_ma)
#     long_ma_list = list(set(long_ma_list))
#
#     results_momentum = []
#     results_reversion = []
#
#     # for short_ma, long_ma in tqdm(ma_combinations, desc='Testing MA combinations'):
#     #     bt_m = MovingAverageBacktester(data,
#     #         strategy='crossover',
#     #         short_window=short_ma,
#     #         long_window=long_ma,
#     #         mode='momentum')
#     #     bt_m.apply_strategy()
#     #     results_momentum.append(bt_m.results())
#     #
#     #     bt_r = MovingAverageBacktester(data,
#     #         strategy='crossover',
#     #         short_window=short_ma,
#     #         long_window=long_ma,
#     #         mode='reversion')
#     #     bt_r.apply_strategy()
#     #     results_reversion.append(bt_r.results())
#
#     for short_ma in tqdm(short_ma_list, desc='Testing MA combinations'):
#         for long_ma in long_ma_list:
#             if short_ma >= long_ma:
#                 continue
#
#             bt_m = MovingAverageBacktester(
#                 data, strategy='crossover',
#                 short_window=short_ma,
#                 long_window=long_ma,
#                 mode='momentum'
# )
#             bt_m.apply_strategy()
#             results_momentum.append(bt_m.results())
#
# #             bt_r = MovingAverageBacktester(
# #                 data, strategy='crossover',
# #                 short_window=short_ma,
# #                 long_window=long_ma,
# #                 mode='reversion'
# # )
# #             bt_r.apply_strategy()
# #             results_reversion.append(bt_r.results())
#
#     df_momentum = pd.DataFrame(results_momentum)
#     # df_reversion = pd.DataFrame(results_reversion)
#
#     print("\n📈 Top 5 Momentum Strategies by Sharpe Ratio:")
#     print(df_momentum.sort_values(by='Sharpe Ratio', ascending=False)[['Short MA', 'Long MA', 'Sharpe Ratio', 'Return (%)', 'Max Drawdown (%)', 'Calmar Ratio']].head(5))
#
#     # print("\n📉 Top 5 Reversion Strategies by Sharpe Ratio:")
#     # print(df_reversion.sort_values(by='Sharpe Ratio', ascending=False)[['Short MA', 'Long MA', 'Sharpe Ratio', 'Return (%)', 'Max Drawdown (%)', 'Calmar Ratio']].head(5))
#
#     plot_heatmap(df_momentum, 'Sharpe Ratio', 'Momentum Strategy - Sharpe Ratio')
#     # plot_heatmap(df_reversion, 'Sharpe Ratio', 'Reversion Strategy - Sharpe Ratio')
#
#     plot_strategy_vs_benchmark(data, short_ma=8, long_ma=32, mode='momentum')
#     plot_strategy_vs_benchmark(data, short_ma=5, long_ma=20, mode='reversion')
#
#
#
    # import plotly.express as px
    # fig = px.line(df, x=df.index, y=['z'], title='')
    # fig.show()