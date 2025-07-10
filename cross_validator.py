
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from backtest_engine import BacktestEngine


class CrossValidator:
    """
    This class is responsible for evaluating backtest results
    and assessing model performance using cross-validation.
    """

    def __int__(self):
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
        pass
        summary = {
            'window': task['x'],
            'y': task['y'],
            'sharpe': task['backtest_result']['result'][0]['sharpe'],
            'factor_currency': task['factor_currency'],
            'asset_currency': task['asset_currency'],
            'timeframe': task['timeframe'],
            'endpoint': task['endpoint'],
        }
        summary.update(model_scores)
        return summary

    @staticmethod
    def generate_features_and_target(action, backtest_df, indicator, orientation, x, y, **kwargs):
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
        else:
            raise ValueError(f"Unsupported indicator: {indicator}")

        backtest_df = BacktestEngine.compute_position(**params)
        if indicator == 'bband':
            features = backtest_df[['factor', 'price', 'upper_band', 'lower_band', 'z']].iloc[x:]
        else:
            features = backtest_df[['price', 'rsi']].iloc[x:]

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