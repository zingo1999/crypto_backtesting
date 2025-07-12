from typing import Tuple

import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from backtest_engine import BacktestEngine
from crypto_data_service import CryptoDataService
from crypto_exchange_data_service import CryptoExchangeDataService
from utilities import Utilities


class CrossValidator:
    """
    This class is responsible for evaluating backtest results
    and assessing model performance using cross-validation.
    """

    def __init__(self):
        pass

    @classmethod
    def evaluate_backtest_results(cls, task, **kwargs):
        """
        Evaluate backtest results and assess model performance.

        Parameters:
        - task: A dictionary containing task details including features and target variables.

        Returns:
        - A summary dictionary with evaluation metrics and model scores.
        """
        features, target = cls.generate_features_and_target(**task)
        model_scores = cls.assess_model_performance(features, target)
        summary = {
            # 'factor_currency': task['factor_currency'],
            # 'asset_currency': task['asset_currency'],
            # 'timeframe': task['timeframe'],
            # 'x': task['x'],
            # 'y': task['y'],
            # 'sharpe': task['backtest_result']['sharpe'],
        }
        summary.update(model_scores)
        summary.update({'strategy': task['backtest_result']['strategy'],})
        return summary

    @staticmethod
    def generate_features_and_target(action: str, backtest_df: pd.DataFrame, indicator: str, orientation: str, x: int, y: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate features and target variables based on the specified indicator.

        Parameters:
        - action: Action to be performed.
        - backtest_df: DataFrame containing backtest data.
        - indicator: Indicator type ('bband' or 'rsi').
        - orientation: Orientation for features.
        - x: Window size for calculations.
        - y: Target variable.

        Returns:
        - A tuple of features and target variables.
        """
        params = {
            'action': action,
            'indicator': indicator,
            'orientation': orientation,
            'x': x,
            'y': y,
            'df': backtest_df,
        }
        if indicator == 'bband':
            backtest_df['ma'] = backtest_df['factor'].rolling(x).mean()
            backtest_df['std'] = backtest_df['factor'].rolling(x).std()
            backtest_df['z'] = (backtest_df['factor'] - backtest_df['ma']) / backtest_df['std']
            backtest_df['upper_band'] = backtest_df['ma'] + 2 * backtest_df['std']
            backtest_df['lower_band'] = backtest_df['ma'] - 2 * backtest_df['std']
        elif indicator == 'rsi':
            delta = backtest_df['factor'].diff(1).fillna(0)
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(x).mean()
            avg_loss = loss.rolling(x).mean()
            rs = avg_gain / avg_loss
            backtest_df['rsi'] = 100 - (100 / (1 + rs))
        else: raise ValueError(f"Unsupported indicator: {indicator}")
        backtest_df = BacktestEngine.compute_position(**params)
        if indicator == 'bband': features = backtest_df[['factor', 'price', 'upper_band', 'lower_band', 'z']].iloc[x:]
        else: features = backtest_df[['price', 'rsi']].iloc[x:]
        backtest_df['trade'] = (backtest_df['pos'].diff().abs() > 0).astype(int)
        backtest_df['pnl'] = backtest_df['pos'].shift(1) * backtest_df['chg'] - backtest_df['trade'] * 0.06 / 100
        backtest_df['cumulative_pnl'] = backtest_df['pnl'].cumsum()
        target = backtest_df['cumulative_pnl'].iloc[x:]
        return features, target

    @staticmethod
    def assess_model_performance(features, target):
        """
        Assess the performance of regression models using cross-validation.

        Parameters:
        - features: Feature DataFrame.
        - target: Target variable Series.

        Returns:
        - A dictionary of model performance metrics (e.g., mean MSE).
        """
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42)
        }
        results = {}
        for name, model in models.items():
            mean_mse = CrossValidator.cross_validate_model(model, features, target)
            results[f'{name}_mean_mse'] = mean_mse
        return results

    @staticmethod
    def cross_validate_model(model, features, target, n_splits=5):
        """
        Perform cross-validation on the given model.

        Parameters:
        - model: Model to be evaluated.
        - features: Feature DataFrame.
        - target: Target variable Series.
        - n_splits: Number of cross-validation splits.

        Returns:
        - Mean squared error across all splits.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mse_scores = []
        for train_idx, test_idx in tscv.split(features):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)
        return np.mean(mse_scores)

    @classmethod
    def generate_tasks_from_results(cls, backtest_dataframe_keys: list, filtered_results: list, minimum_sharpe: float, kwargs: dict) -> pd.DataFrame:
        backtest_dataframes = {
            backtest_key: CryptoDataService({
                **kwargs,
                'asset_currency': backtest_key.split('|')[1],
                'factor_currency': backtest_key.split('|')[0],
                'timeframe': backtest_key.split('|')[2],
                'endpoint': backtest_key.split('|')[3]
            }).create_backtest_dataframe(CryptoExchangeDataService(backtest_key.split('|')[1], **kwargs).get_historical_data(True)) for backtest_key in backtest_dataframe_keys
        }
        tasks = [{
            'action': data['action'],
            'backtest_df': backtest_dataframes[data['backtest_dataframe_key']].copy(),
            'backtest_result': data['result'],
            'indicator': data['indicator'],
            'minimum_sharpe': minimum_sharpe,
            'orientation': data['orientation'],
            'timeframe': data['timeframe'],
            'x': int(data['result']['x']),
            'y': float(data['result']['y']),
        } for data in filtered_results]

        # if tasks:
        #     if len(tasks) > 1:
        #         cross_validation_results = Utilities.run_in_parallel(CrossValidator.evaluate_backtest_results, tasks)
        #     else:
        #         cross_validation_results = [CrossValidator.evaluate_backtest_results(tasks[0])]
        #     cv_results_df = pd.DataFrame(cross_validation_results)
        #     print(cv_results_df.head())
        #     return cv_results_df

        if tasks:
            cross_validation_result = (Utilities.run_in_parallel(CrossValidator.evaluate_backtest_results, tasks) if len(tasks) > 1 else [CrossValidator.evaluate_backtest_results(tasks[0])])
            return cross_validation_result