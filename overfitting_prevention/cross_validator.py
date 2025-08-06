import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from analysis_tools.crypto_data_service import CryptoDataService
from analysis_tools.crypto_exchange_data_service import CryptoExchangeDataService
from analysis_tools.utilities import Utilities


class CrossValidator:
    """
    This class is responsible for evaluating backtest results
    and assessing model performance using cross-validation.
    """
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

    def __init__(self, asset_currency, kwargs):
        self.asset_currency_list = ['BTC', 'ETH', 'SOL'] if not asset_currency else [asset_currency]
        self.kwargs = kwargs
        self.kwargs['update_mode'] = False

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
            **model_scores,
            'strategy': task['strategy']
        }
        return summary

    @staticmethod
    def generate_features_and_target(backtest_df: pd.DataFrame, indicator: str, x: int, **kwargs):
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
        if indicator == 'bband':
            backtest_df['ma'] = backtest_df['factor'].rolling(x).mean()
            backtest_df['std'] = backtest_df['factor'].rolling(x).std()
            backtest_df['z'] = (backtest_df['factor'] - backtest_df['ma']) / backtest_df['std']

        elif indicator == 'rsi':
            backtest_df['delta'] = backtest_df['factor'].diff(1)
            backtest_df['delta'] = backtest_df['delta'].astype(float).fillna(0)
            backtest_df['positive'] = backtest_df['delta'].clip(lower=0)
            backtest_df['negative'] = backtest_df['delta'].clip(upper=0)
            backtest_df['average_gain'] = backtest_df['positive'].rolling(x).mean()
            backtest_df['average_loss'] = abs(backtest_df['negative'].rolling(x).mean())
            backtest_df['relative_strength'] = backtest_df['average_gain'] / backtest_df['average_loss']
            backtest_df['z'] = 100 - (100 / (1 + backtest_df['relative_strength']))

        else: raise ValueError(f"Unsupported indicator: {indicator}")

        features = backtest_df[['factor', 'z']].iloc[x:]
        target = backtest_df['chg'].shift(-1).iloc[x:]

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
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=5,
                max_features=2,
                # max_features='sqrt',
                # max_features='log2',
                random_state=42)
        }
        results = {}
        for name, model in models.items():
            mean_mse = CrossValidator.cross_validate_model(model, features, target)
            results[f'{name}_mean_mse'] = round(mean_mse * 1000, 2)
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
        features = features.copy()
        features = features.replace([np.inf, -np.inf], np.nan)

        mse_scores = []
        for train_idx, test_idx in tscv.split(features):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mask = ~np.isnan(y_test)
            mse = mean_squared_error(y_test[mask], y_pred[mask])
            mse_scores.append(mse)
        return np.mean(mse_scores)

    @classmethod
    def run_cross_validation(cls, result_df, kwargs):
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
        for _, row in result_df.iterrows():
            backtest_dataframe_key = '|'.join(row['strategy'].rsplit('|', 5)[:-5])
            backtest_df = backtest_dataframe_map[backtest_dataframe_key].copy()
            tasks.append({
                'action': row['strategy'].split('|')[-3],
                'backtest_df': backtest_df,
                'indicator': row['strategy'].split('|')[-5],
                'orientation': row['strategy'].split('|')[-4],
                'sharpe': row['sharpe'],
                'strategy': row['strategy'],
                'timeframe': row['strategy'].split('|')[2],
                'x': row['x'],
                'y': row['y'],
            })
        if tasks:
            cross_validation_result = (Utilities.run_in_parallel(CrossValidator.evaluate_backtest_results, tasks) if len(tasks) > 1 else [CrossValidator.evaluate_backtest_results(tasks[0])])
            return pd.DataFrame(cross_validation_result)

    def process_backtest_results(self):
        for asset_currency in self.asset_currency_list:
            self.kwargs['asset_currency'] = asset_currency
            file_path = f"backtest_results/{asset_currency}/{self.kwargs['since']}/{asset_currency}_filtered_result.csv"
            if os.path.exists(file_path):
                result_df = pd.read_csv(file_path, index_col=0).query('parameter_plateau').dropna(subset=['training_set', 'testing_set', 'parameter_plateau'])
                result_df = result_df[result_df['sharpe'] > self.kwargs['minimum_sharpe']].reset_index(drop=True)
                if result_df.empty: return
                cv_result_df = self.run_cross_validation(result_df, self.kwargs)
                if cv_result_df.empty: return
                cv_columns = cv_result_df.columns[:-1]
                result_df = result_df.drop(columns=cv_columns, errors='ignore')

                result_df = pd.merge(result_df, cv_result_df, on='strategy', how='inner')
                cols = [col for col in result_df.columns if col != 'strategy'] + ['strategy']
                result_df = result_df[cols]
                result_df = result_df.sort_values(by='sharpe', ascending=False).reset_index(drop=True)

                file_path = f"backtest_results/{asset_currency}/{self.kwargs['since']}/{asset_currency}_optimized_result.csv"
                result_df.to_csv(file_path)
                print(result_df.head())



