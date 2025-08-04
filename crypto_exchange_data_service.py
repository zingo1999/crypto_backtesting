# Standard library imports
import logging
import datetime
import os
import sys
import time
from typing import Dict, List, Optional, Union

# Third-party imports
import ccxt
import requests
import pandas as pd
# from ccxt.base.exchange import Exchange

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

logging.basicConfig(level=logging.INFO)

data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)


# Class definition
class CryptoExchangeDataService:
    # Class-level constant variables representing fixed data
    EXCHANGE_NAMES: List[str]                   = ['binance', 'binanceusdm', 'bitget', 'bitmart', 'bybit', 'coinbase', 'cryptocom',
                                 'deribit', 'gate', 'hashkey', 'huobi', 'kucoin', 'kucoinfutures', 'kraken',
                                 'krakenfutures', 'lbank', 'mexc', 'okx', 'tokocrypto', 'upbit', 'xt']
    FUNDING_RATE_TIMEFRAME: str                 = '8h'
    OI_INTERVAL_MAP: Dict[str, str]             = {
        '1m': '5m',
        '3m': '5m',
        '5m': '5m',
        '10m': '5m',
        '15m': '15m',
        '30m': '30m',
        '60m': '1h',
        '1h': '1h',
        '2h': '1h',
        '4h': '4h',
        '6h': '4h',
        '8h': '4h',
        '12h': '4h',
        '24h': '1d',
        '1d': '1d',
    }
    RESAMPLING_FREQUENCY_MAP: Dict[str, str]    = {
        '1m': '1min',  # 1 minute
        '2m': '2min',  # 2 minutes
        '3m': '3min',  # 3 minutes
        '5m': '5min',  # 5 minutes
        '10m': '10min',  # 10 minutes
        '15m': '15min',  # 15 minutes
        '30m': '30min',  # 30 minutes
        '1h': 'h',  # 1 hour
        '2h': '2h',
        '3h': '3h',
        '4h': '4h',
        '6h': '6h',
        '8h': '8h',
        '12h': '12h',
        '24h': 'D',
        '1d': 'D',
    }
    TIMEFRAME_LABEL_MAP: Dict[str, str]         = {
        '24h': 'D',
        '1d': 'D',
        '12h': '12h',
        '6h': '6h',
        '4h': '4h',
        '2h': '2h',
        '1h': 'h',
        '30m': '30min',
        '15m': '15min',
        '5m': '5min',
        '1m': '1min', }
    TIMEFRAME_MAP: Dict[str, str]               = {
        '1m': '1m',
        '3m': '3m',
        '5m': '5m',
        '10m': '5m',
        '15m': '15m',
        '30m': '30m',
        '60m': '1h',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '8h': '8h',
        '12h': '12h',
        '24h': '1d',
        '1d': '1d',
    }
    TIMEFRAME_TO_MS: Dict[str, int]             = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '8h': 8 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }
    TIMESTAMP_MS_THRESHOLD: int                 = 1_000_000_000_000

    # Class-level mutable variables
    pass

    def __init__(self, **kwargs):
        """Initializes instance variables of the class."""

        # Set default values for instance variables
        self.asset_currency = ''
        self.endpoint       = ''
        self.exchange_name  = ''
        self.product_type   = ''
        self.timeframe      = ''
        self.since          = ''
        self.update_mode    = False

        # Update instance variables with provided keyword arguments
        for key, value in kwargs.items(): setattr(self, key, value)

        if self.exchange_name not in self.EXCHANGE_NAMES: raise ValueError(f"Exchange '{self.exchange_name}' is not supported.")
        self.exchange = getattr(ccxt, self.exchange_name)()
        self.rate_limit_seconds = self.exchange.rateLimit / 1000
        self.endpoint_methods = {
            'funding_rate': self.exchange.fetch_funding_rate_history,
            'index_price': self.exchange.fetch_index_ohlcv,
            'long_short_ratio': self.exchange.fetch_long_short_ratio_history,
            'mark_price': self.exchange.fetch_mark_ohlcv,
            'open_interest': self.exchange.fetch_open_interest_history,
            'premium_index': self.exchange.fetch_premium_index_ohlcv,
            'price': self.exchange.fetch_ohlcv, }


    ##### INSTANCE METHOD SECTION #####
    """instance methods are used to access or modify the attributes of the instance"""
    def get_historical_data(self, underlying_asset=False) -> pd.DataFrame:
        def extract_candle_data_as_dataframe(candle_list: list, endpoint) -> pd.DataFrame:
            """
            Converts a list of candle data into a DataFrame, handling specific data types and ensuring unique timestamps.
            """

            if endpoint in ['funding_rate', 'open_interest', 'long_short_ratio']:
                # key_index = 2 if endpoint in ['funding_rate', 'open_interest'] else 5
                if endpoint == 'open_interest': key_index = 1
                elif endpoint == 'funding_rate': key_index = 2
                else: key_index = 5
                endpoint = list(candle_list[0].keys())[key_index]
                data = {'timestamp': [candle['timestamp'] for candle in candle_list], endpoint: [candle[endpoint] for candle in candle_list]}
                df = pd.DataFrame(data).rename(columns={'timestamp': 'unix_timestamp'})
            else:
                df = pd.DataFrame.from_records(candle_list)

            if df.shape[1] == 6: df = df.rename(columns={
                0: 'unix_timestamp',
                1: 'open',
                2: 'high',
                3: 'low',
                4: 'close',
                5: 'volume', })
            df = (df.assign(datetime=lambda x: pd.to_datetime(x['unix_timestamp'], unit='ms')).drop_duplicates(subset='unix_timestamp', keep='last').sort_values('unix_timestamp').set_index('datetime').dropna(axis=1, how='all'))

            return df.dropna()

        def resample_data(df, endpoint):
            resample_freq = self.RESAMPLING_FREQUENCY_MAP.get(self.timeframe, 'h')
            if endpoint == 'price': df = df.resample(resample_freq).agg({
                    'unix_timestamp': 'first',  # Take the last timestamp in the period
                    'open': 'first',  # Take the first open price in the period
                    'high': 'max',  # Take the maximum high price in the period
                    'low': 'min',  # Take the minimum low price in the period
                    'close': 'last',  # Take the last close price in the period
                    'volume': 'sum',  # Sum the volume for the period
                }).iloc[:-1]
            elif endpoint == 'open_interest': df = df.resample(resample_freq, label='left', closed='left').agg({
                    'unix_timestamp': 'last',
                    'openInterestAmount': 'first',
                })
            elif endpoint == 'long_short_ratio':
                pass
            elif endpoint == 'premium_index': df = df.resample(resample_freq).agg({
                    'unix_timestamp': 'first',  # Last timestamp in the hour
                    'open': 'first',           # First open price in the hour
                    'high': 'max',             # Maximum high price in the hour
                    'low': 'min',              # Minimum low price in the hour
                    'close': 'last',           # Last close price in the hour
                }).iloc[:-1]

            return df


        # Initialize parameters for fetching financial data from an exchange, including data types, file formats, and timestamps.
        endpoint = 'price' if underlying_asset is True else self.endpoint
        subfolder_path = os.path.join(data_folder, 'exchange')
        os.makedirs(subfolder_path, exist_ok=True)
        file_formats = ['pkl', 'parquet', 'csv']
        fmt, saved_df = None, None
        key = 'timestamp' if endpoint in ['funding_rate', 'open_interest', 'long_short_ratio'] else 0
        product_type = 'linear' if endpoint in ['funding_rate', 'open_interest', 'long_short_ratio'] else self.product_type

        asset_currency = self.asset_currency.upper()
        timeframe = '1m'
        symbol = f"{asset_currency}/USDT:USDT" if product_type == 'linear' else f"{asset_currency}/USDT"
        params = {'symbol': symbol}

        if endpoint == 'open_interest': timeframe = self.OI_INTERVAL_MAP.get(timeframe)
        elif endpoint == 'long_short_ratio' and timeframe not in ['1h', '1d']: timeframe = '1h'
        if endpoint == 'funding_rate': timeframe = self.FUNDING_RATE_TIMEFRAME
        else: params.update({'timeframe': timeframe})
        base_path = os.path.join(subfolder_path, f"{asset_currency}|{self.exchange_name}")
        data_path = f"{base_path}|{product_type}|{timeframe}|{endpoint}"

        if isinstance(self.since, int) and self.since < self.TIMESTAMP_MS_THRESHOLD: requested_start_timestamp = int(self.since * 1000)
        elif isinstance(self.since, str): requested_start_timestamp = int(pd.to_datetime(self.since).timestamp()) * 1000
        else: requested_start_timestamp = self.since

        # Fetch and combine candle data from the exchange and existing files into a DataFrame.
        all_candles = []
        try:
            latest_candle_data = self.endpoint_methods[endpoint](**params)
            time.sleep(self.rate_limit_seconds)
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise
        all_candles += latest_candle_data
        candle_count = len(latest_candle_data)
        previous_time_interval_ms = self.TIMEFRAME_TO_MS[timeframe] * candle_count

        first_fetched_timestamp = latest_candle_data[0][key]
        last_fetched_timestamp = latest_candle_data[-1][key]
        last_marked_timestamp = latest_candle_data[-1][key]

        for fmt in file_formats:
            file_path = f"{data_path}.{fmt}"
            if os.path.isfile(file_path):
                load_func = {
                    'pkl': pd.read_pickle,
                    'parquet': pd.read_parquet,
                    'csv': pd.read_csv}[fmt]
                try:
                    saved_df = load_func(file_path)
                except Exception as e:
                    print(f"{e}")
                first_fetched_timestamp = min(saved_df['unix_timestamp'].iloc[0], first_fetched_timestamp)
                last_marked_timestamp = min(saved_df['unix_timestamp'].iloc[-1], last_marked_timestamp)
                break
        fmt = fmt if saved_df is not None else 'pkl'

        if self.update_mode is True:
            while requested_start_timestamp < first_fetched_timestamp:
                new_fetch_since_timestamp = first_fetched_timestamp - previous_time_interval_ms
                params.update({'since': new_fetch_since_timestamp})
                try:
                    fetched_candle_data = self.endpoint_methods[endpoint](**params)
                    time.sleep(self.rate_limit_seconds)
                except Exception as e:
                    logging.error(f"Error fetching data: {e}")
                    raise
                if not fetched_candle_data or fetched_candle_data[0][key] > new_fetch_since_timestamp: break
                all_candles += fetched_candle_data
                first_fetched_timestamp = fetched_candle_data[0][key]

            while last_marked_timestamp < last_fetched_timestamp:
                new_fetch_since_timestamp = last_marked_timestamp
                params.update({'since': new_fetch_since_timestamp})
                try:
                    fetched_candle_data = self.endpoint_methods[endpoint](**params)
                    time.sleep(self.rate_limit_seconds)
                except Exception as e:
                    logging.error(f"Error fetching data: {e}")
                    raise
                all_candles += fetched_candle_data
                last_marked_timestamp = fetched_candle_data[-1][key]

            new_data_df = extract_candle_data_as_dataframe(all_candles, endpoint)
            if saved_df is not None and not saved_df.empty:
                df = pd.concat([new_data_df, saved_df], ignore_index=True)
                df = df.assign(datetime=lambda x: pd.to_datetime(x['unix_timestamp'], unit='ms'))
                df = df.drop_duplicates(subset='unix_timestamp', keep='last').sort_values('unix_timestamp').set_index('datetime').dropna()
            else: df = new_data_df

            if endpoint not in ['funding_rate', 'premium_index', 'long_short_ratio']: df = df.iloc[:-1]

            file_saver = {
                'pkl': df.to_pickle,
                'csv': df.to_csv,
                'parquet': df.to_parquet}
            file_saver[fmt](f"{data_path}.{fmt}")

        else:
            if saved_df is not None and not saved_df.empty: df = saved_df
            else: df = extract_candle_data_as_dataframe(latest_candle_data, endpoint)

        # Check for missing timestamps in the DataFrame and save the data to a specified file format.
        start_time = pd.to_datetime(df['unix_timestamp'].iloc[0], unit='ms')
        end_time = pd.to_datetime(df['unix_timestamp'].iloc[-1], unit='ms')
        freq = self.TIMEFRAME_LABEL_MAP.get(timeframe, 'D')
        time_range = pd.date_range(start=start_time, end=end_time, freq=freq)
        missing_times = time_range[~time_range.isin(df.index)]
        if len(missing_times) > 0: logging.warning(f"Missing {len(missing_times)} {endpoint.replace('_', ' ')} data found\n{missing_times}")
        else: logging.info(f"No missing {endpoint.replace('_', ' ')} data\n{df.index[0]} to {df.index[-1]}")

        df = df[df['unix_timestamp'] >= requested_start_timestamp]
        df = resample_data(df, endpoint)

        return df


    ##### CLASS METHOD SECTION #####
    """class methods can be used to access or modify class state (i.e., class variables) or to create factory methods"""

    @classmethod
    def update_class_constants(cls):
        pass

    @classmethod
    def create_instance_from_config(cls):
        pass


    ##### STATIC METHOD SECTION #####
    """static methods are used to perform utility functions that are related to the class but do not need access to class or instance attributes"""

    @staticmethod
    def utility_function():
        pass


class ConvertCcxtDataSerice:
    # Class-level constant variables representing fixed data

    @classmethod
    def get_historical_data(cls, asset_currency, endpoint, exchange_name, product_type, timeframe, underlying_asset=False, **kwargs):

        oi_interval_map: Dict[str, str] = {
            '1m': '5m',
            '3m': '5m',
            '5m': '5m',
            '10m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '1h',
            '1h': '1h',
            '2h': '1h',
            '4h': '4h',
            '6h': '4h',
            '8h': '4h',
            '12h': '4h',
            '24h': '1d',
            '1d': '1d',
            '7d': '1d',
            '1w': '1d',
            '1M': '1d',
            '1month': '1d'}
        timeframe_label_map: Dict[str, str] = {
            '24h': 'D',
            '1d': 'D',
            '12h': '12h',
            '6h': '6h',
            '4h': '4h',
            '2h': '2h',
            '1h': 'h',
            '30m': '30min',
            '15m': '15min',
            '5m': '5min',
            '1m': '1min', }
        timeframe_map: Dict[str, str] = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '10m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '1h',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '24h': '1d',
            '1d': '1d',
            '7d': '1w',
            '1w': '1w',
            '1M': '1M',
            '1month': '1M'}
        timeframe_to_ms: Dict[str, int] = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000, }
        exchange_names: List[str] = ['binance', 'binanceusdm', 'bitget', 'bitmart', 'bybit', 'coinbase', 'cryptocom',
                                     'deribit', 'gate', 'hashkey', 'huobi', 'kucoin', 'kucoinfutures', 'kraken',
                                     'krakenfutures', 'lbank', 'mexc', 'okx', 'tokocrypto', 'upbit', 'xt']
        timestamp_ms_threshold: int = 1_000_000_000_000
        funding_rate_timeframe: str = '8h'

        if exchange_name not in exchange_names: raise ValueError(f"Exchange '{exchange_name}' is not supported.")
        exchange = getattr(ccxt, exchange_name)()
        rate_limit_seconds = exchange.rateLimit / 1000
        endpoint_methods = {
            'funding_rate': exchange.fetch_funding_rate_history,
            'index_price': exchange.fetch_index_ohlcv,
            'long_short_ratio': exchange.fetch_long_short_ratio_history,
            'mark_price': exchange.fetch_mark_ohlcv,
            'open_interest': exchange.fetch_open_interest_history,
            'premium_index': exchange.fetch_premium_index_ohlcv,
            'price': exchange.fetch_ohlcv, }

        # Initialize parameters for fetching financial data from an exchange, including data types, file formats, and timestamps.
        if underlying_asset is True: endpoint = 'price'
        subfolder_path = os.path.join(data_folder, 'exchange')
        os.makedirs(subfolder_path, exist_ok=True)
        file_formats = ['pkl', 'parquet', 'csv']
        fmt, saved_df = None, None
        key = 'timestamp' if endpoint in ['funding_rate', 'open_interest', 'long_short_ratio'] else 0
        if endpoint in ['funding_rate', 'open_interest', 'long_short_ratio']: product_type = 'linear'

        asset_currency = asset_currency.upper()
        timeframe = timeframe_map[timeframe]
        symbol = f"{asset_currency}/USDT:USDT" if product_type == 'linear' else f"{asset_currency}/USDT"

        params = {'symbol': symbol}

        if endpoint == 'open_interest': timeframe = oi_interval_map.get(timeframe)
        elif endpoint == 'long_short_ratio' and timeframe not in ['1h', '1d']:
            logging.error("(line 424) Only accept '1h' and '1d' interval/timeframe")
            sys.exit()
        if endpoint == 'funding_rate': timeframe = funding_rate_timeframe
        else: params.update({'timeframe': timeframe})
        base_path = os.path.join(subfolder_path, f"{asset_currency}|{exchange_name}")
        data_path = f"{base_path}|{product_type}|{timeframe}|{endpoint}"

        if isinstance(since, int) and since < timestamp_ms_threshold: requested_start_timestamp = int(since * 1000)
        elif isinstance(since, str): requested_start_timestamp = int(pd.to_datetime(since).timestamp()) * 1000
        else: requested_start_timestamp = since

        # Fetch and combine candle data from the exchange and existing files into a DataFrame.
        all_candles = []
        try:
            latest_candle_data = endpoint_methods[endpoint](**params)
            time.sleep(rate_limit_seconds)
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise
        all_candles += latest_candle_data
        candle_count = len(latest_candle_data)
        previous_time_interval_ms = timeframe_to_ms[timeframe] * candle_count

        first_fetched_timestamp = latest_candle_data[0][key]
        last_fetched_timestamp = latest_candle_data[-1][key]
        last_marked_timestamp = latest_candle_data[-1][key]

        for fmt in file_formats:
            file_path = f"{data_path}.{fmt}"
            if os.path.isfile(file_path):
                load_func = {
                    'pkl': pd.read_pickle,
                    'parquet': pd.read_parquet,
                    'csv': pd.read_csv}[fmt]
                saved_df = load_func(file_path)
                first_fetched_timestamp = min(saved_df['unix_timestamp'].iloc[0], first_fetched_timestamp)
                last_marked_timestamp = min(saved_df['unix_timestamp'].iloc[-1], last_marked_timestamp)
                break
        fmt = fmt if saved_df is not None else 'parquet'

        while requested_start_timestamp < first_fetched_timestamp:
            new_fetch_since_timestamp = first_fetched_timestamp - previous_time_interval_ms
            params.update({'since': new_fetch_since_timestamp})
            try:
                fetched_candle_data = endpoint_methods[endpoint](**params)
                time.sleep(rate_limit_seconds)
            except Exception as e:
                logging.error(f"Error fetching data: {e}")
                raise
            if not fetched_candle_data or fetched_candle_data[0][key] > new_fetch_since_timestamp: break
            all_candles += fetched_candle_data
            first_fetched_timestamp = fetched_candle_data[0][key]

        while last_marked_timestamp < last_fetched_timestamp:
            new_fetch_since_timestamp = last_marked_timestamp
            params.update({'since': new_fetch_since_timestamp})
            try:
                fetched_candle_data = endpoint_methods[endpoint](**params)
                time.sleep(rate_limit_seconds)
            except Exception as e:
                logging.error(f"Error fetching data: {e}")
                raise
            all_candles += fetched_candle_data
            last_marked_timestamp = fetched_candle_data[-1][key]

        new_data_df = cls.extract_candle_data_as_dataframe(all_candles, endpoint)
        if saved_df is not None and not saved_df.empty:
            df = pd.concat([new_data_df, saved_df], ignore_index=True)
            df = df.assign(datetime=lambda x: pd.to_datetime(x['unix_timestamp'], unit='ms'))
            df = df.drop_duplicates(subset='unix_timestamp', keep='last').sort_values('unix_timestamp').set_index('datetime').dropna()
        else: df = new_data_df

        if endpoint not in ['funding_rate', 'open_interest', 'premium_index', 'long_short_ratio']: df = df.iloc[:-1]
        df = df[df['unix_timestamp'] >= requested_start_timestamp]

        # Check for missing timestamps in the DataFrame and save the data to a specified file format.
        start_time = pd.to_datetime(df['unix_timestamp'].iloc[0], unit='ms')
        end_time = pd.to_datetime(df['unix_timestamp'].iloc[-1], unit='ms')
        freq = timeframe_label_map.get(timeframe, 'D')
        time_range = pd.date_range(start=start_time, end=end_time, freq=freq)
        missing_times = time_range[~time_range.isin(df.index)]
        if len(missing_times) > 0:  logging.warning(f"Missing {len(missing_times)} {endpoint.replace('_', ' ')} data found\n{missing_times}")
        else: logging.info(f"No missing {endpoint.replace('_', ' ')} data\n{df.index[0]} to {df.index[-1]}")

        file_saver = {
            'pkl': df.to_pickle,
            'csv': df.to_csv,
            'parquet': df.to_parquet}
        file_saver[fmt](f"{data_path}.{fmt}")

        return df

    @staticmethod
    def extract_candle_data_as_dataframe(candle_list: list, endpoint) -> pd.DataFrame:
        """
        Converts a list of candle data into a DataFrame, handling specific data types and ensuring unique timestamps.
        """
        if endpoint in ['funding_rate', 'open_interest', 'long_short_ratio']:
            key_index = 2 if endpoint in ['funding_rate', 'open_interest'] else 5
            endpoint = list(candle_list[0].keys())[key_index]
            data = {'timestamp': [candle['timestamp'] for candle in candle_list], endpoint: [candle[endpoint] for candle in candle_list]}
            df = pd.DataFrame(data).rename(columns={'timestamp': 'unix_timestamp'})
        else:
            df = pd.DataFrame.from_records(candle_list)
        if df.shape[1] == 6: df = df.rename(columns={
            0: 'unix_timestamp',
            1: 'open',
            2: 'high',
            3: 'low',
            4: 'close',
            5: 'volume', })
        df = (df.assign(datetime=lambda x: pd.to_datetime(x['unix_timestamp'], unit='ms')).drop_duplicates(subset='unix_timestamp', keep='last').sort_values('unix_timestamp').set_index('datetime').dropna(axis=1, how='all'))
        return df.dropna()


# class CryptoBybitDataService:
#     # Class-level constant variables representing fixed data
#     BASE_URL = 'https://api.bybit.com/v5/'
#     ENDPOINT_PATH_MAP: Dict[str, str]           = {
#         'price': 'market/kline',
#         'mark_price': 'market/mark-price-kline',
#         'index_price': 'market/index-price-kline',
#         'premium_index': 'premium-index-price-kline',
#         'instruments_info': 'market/instruments-info',
#         'orderbook': 'market/orderbook',
#         'tickers': 'market/tickers',
#         'funding_rate': 'market/funding/history',
#         'open_interest': 'market/open-interest',
#         'historical_volatility': 'market/historical-volatility',
#         'insurance': 'market/insurance',
#         'risk_limit': 'market/risk-limit',
#         'delivery_price': 'market/delivery-price',
#         'long_short_ratio': 'market/account-ratio',
#
#     }
#     OI_INTERVAL_MAP: Dict[str, str]             = {
#         1: '5m',
#         '1m': '5m',
#         3: '5m',
#         '3m': '5m',
#         5: '5m',
#         '5m': '5m',
#         10: '5m',
#         '10m': '5m',
#         15: '15m',
#         '15m': '15m',
#         30: '30m',
#         '30m': '30m',
#         60: '1h',
#         '60m': '1h',
#         '1h': '1h',
#         120: '1h',
#         '2h': '1h',
#         240: '4h',
#         '4h': '4h',
#         360: '4h',
#         '6h': '4h',
#         480: '4h',
#         '8h': '4h',
#         720: '4h',
#         '12h': '4h',
#         1440: '1d',
#         '24h': '1d',
#         '1d': '1d',
#         '7d': '1d',
#         '1w': '1d',
#         '1M': '1d',
#         '1month': '1d'}
#     TIMEFRAME_MAP: Dict[Union[str, int], str]   = {
#         '1m': 1,
#         '3m': 3,
#         '5m': 5,
#         '10m': 5,
#         '15m': 15,
#         '30m': 30,
#         60: 60,
#         '1h': 60,
#         120: 120,
#         '2h': 120,
#         240: 240,
#         '4h': 240,
#         360: 360,
#         '6h': 360,
#         '8h': 360,
#         720: 720,
#         '12h': 720,
#         '24h': 'D',
#         '1d': 'D',
#     }
#     TIMEFRAME_TO_MS: Dict[str, int]             = {
#         '1m': 60 * 1000,             1: 60 * 1000,
#         '3m': 3 * 60 * 1000,         3: 3 * 60 * 1000,
#         '5m': 5 * 60 * 1000,         5: 5 * 60 * 1000,
#         '15m': 15 * 60 * 1000,       15: 15 * 60 * 1000,
#         '30m': 30 * 60 * 1000,       30: 30 * 60 * 1000,
#         '1h': 60 * 60 * 1000,       '60m': 60 * 60 * 1000,       60: 60 * 60 * 1000,
#         '120m': 2 * 60 * 60 * 1000,  120: 2 * 60 * 60 * 1000,
#         '240m': 4 * 60 * 60 * 1000,  240: 4 * 60 * 60 * 1000,
#         '360m': 6 * 60 * 60 * 1000,  360: 6 * 60 * 60 * 1000,
#         '480m': 8 * 60 * 60 * 1000,  480: 8 * 60 * 60 * 1000,
#         '720m': 12 * 60 * 60 * 1000, 720: 12 * 60 * 60 * 1000,
#         'D': 24 * 60 * 60 * 1000,
#     }
#     TIMESTAMP_MS_THRESHOLD: int                 = 1_000_000_000_000
#
#
#     # Class-level mutable variables
#
#     pass
#
#
#     ##### Instance variables #####
#     def __init__(self, **kwargs):
#         """Initializes instance variables of the class."""
#         self.asset_currency = 'btc'
#         self.endpoint       = 'price'
#         self.exchange_name  = 'bybit'
#         self.product_type     = 'linear'
#         self.timeframe      = '1m'
#         self.since          = '2020-05-01'
#
#         # Update instance variables with provided keyword arguments
#         for key, value in kwargs.items(): setattr(self, key, value)
#
#
#
#     ##### INSTANCE METHOD SECTION #####
#     """instance methods are used to access or modify the attributes of the instance"""
#     def fetch_historical_data(self,):
#         subfolder_path = os.path.join(data_folder, 'exchange')
#         os.makedirs(subfolder_path, exist_ok=True)
#         file_formats = ['pkl', 'parquet', 'csv']
#         fmt, saved_df = None, None
#
#         asset_currency = self.asset_currency.upper()
#         endpoint_path = self.ENDPOINT_PATH_MAP.get(self.endpoint)
#         path_segment = endpoint_path.split('/')[0]
#         path_segment = path_segment.replace('-', '_')
#
#         endpoint = endpoint_path[len(path_segment) + 1:]
#         endpoint = endpoint.replace('-', '_').replace('/', '_')
#
#
#         key = 'timestamp' if endpoint in ['open_interest', 'long_short_ratio'] else 0
#         if self.endpoint == 'funding_rate':
#             key = 'fundingRateTimestamp'
#
#         symbol = f"{self.asset_currency.upper()}USDT"
#
#         timeframe = self.TIMEFRAME_MAP[self.timeframe]
#         if self.endpoint == 'open_interest': timeframe = self.OI_INTERVAL_MAP.get(self.timeframe)
#
#         since = int(pd.to_datetime(self.since).timestamp()) * 1000
#         if isinstance(self.since, int) and self.since < self.TIMESTAMP_MS_THRESHOLD: requested_start_timestamp = int(self.since * 1000)
#         elif isinstance(self.since, str): requested_start_timestamp = int(pd.to_datetime(self.since).timestamp()) * 1000
#         else: requested_start_timestamp = self.since
#
#         base_path = os.path.join(subfolder_path, f"{asset_currency}|{self.exchange_name}")
#         data_path = f"{base_path}|{product_type}|{timeframe}|{self.endpoint}"
#
#         # meta_df = pd.read_csv('bybit_v5_api.csv')
#         # endpoint_details = meta_df[meta_df['endpoint_path'] == f'/{endpoint_path}'].reset_index(drop=True)
#
#
#         query_params = {
#             'category': self.product_type,
#             'symbol': symbol,
#             # 'interval': timeframe,
#             # 'intervalTime': timeframe,
#             'period': timeframe,
#             # 'startTime': since,
#             # 'limit': 1000,
#         }
#         all_candles = []
#         try:
#             response = requests.get(f"{self.BASE_URL}{endpoint_path}", params=query_params)
#             response.raise_for_status()
#             data = response.json()
#             latest_candle_data = data['result']['list']
#             all_candles += latest_candle_data
#             candle_count = len(latest_candle_data)
#             previous_time_interval_ms = self.TIMEFRAME_TO_MS[timeframe] * candle_count
#
#             first_fetched_timestamp = int(latest_candle_data[0][key])
#             last_fetched_timestamp = int(latest_candle_data[-1][key])
#             last_marked_timestamp = int(latest_candle_data[-1][key])
#
#             for fmt in file_formats:
#                 file_path = f"{data_path}.{fmt}"
#                 if os.path.isfile(file_path):
#                     load_func = {
#                         'pkl': pd.read_pickle,
#                         'parquet': pd.read_parquet,
#                         'csv': pd.read_csv}[fmt]
#                     saved_df = load_func(file_path)
#                     first_fetched_timestamp = min(saved_df['unix_timestamp'].iloc[0], first_fetched_timestamp)
#                     last_marked_timestamp = min(saved_df['unix_timestamp'].iloc[-1], last_marked_timestamp)
#                     break
#             fmt = fmt if saved_df is not None else 'parquet'
#
#             while requested_start_timestamp < first_fetched_timestamp:
#                 new_fetch_since_timestamp = first_fetched_timestamp - previous_time_interval_ms
#                 query_params.update({'since': new_fetch_since_timestamp})
#                 # query_params.update({'startTime': new_fetch_since_timestamp})
#                 try:
#                     response = requests.get(f"{self.BASE_URL}{endpoint_path}", params=query_params)
#                     response.raise_for_status()
#                     data = response.json()
#                     fetched_candle_data = data['result']['list']
#                 except Exception as e:
#                     logging.error(f"Error fetching data: {e}")
#                     raise
#                 if not fetched_candle_data or int(fetched_candle_data[0][key]) > new_fetch_since_timestamp: break
#                 all_candles += fetched_candle_data
#                 first_fetched_timestamp = fetched_candle_data[0][key]
#             pass
#
#
#
#
#             df = pd.DataFrame(data['result']['list'])
#             if df.shape[1] == 6: df = df.rename(columns={
#                     0: 'unix_timestamp',
#                     1: 'open',
#                     2: 'high',
#                     3: 'low',
#                     4: 'close',
#                     5: 'volume', })
#             elif df.shape[1] == 7: df = df.rename(columns={
#                     0: 'unix_timestamp',
#                     1: 'open',
#                     2: 'high',
#                     3: 'low',
#                     4: 'close',
#                     5: 'volume',
#                     6: 'quote_volume',
#             })
#             df = (df
#                   .assign(datetime=lambda x: pd.to_datetime(pd.to_numeric(x['unix_timestamp']), unit='ms'))
#                   .drop_duplicates(subset='unix_timestamp', keep='last')
#                   .sort_values('unix_timestamp')
#                   .set_index('datetime')
#                   .dropna(axis=1, how='all')
#                   )
#             return df.dropna()
#         except requests.exceptions.RequestException as err:
#             logging.error(f"Error fetching data: {err}")
#             return {"error": str(err)}
#
#
#     ##### CLASS METHOD SECTION #####
#     """class methods can be used to access or modify class state (i.e., class variables) or to create factory methods"""
#
#     pass
#
#     ##### STATIC METHOD SECTION #####
#     """static methods are used to perform utility functions that are related to the class but do not need access to class or instance attributes"""
#
#     pass

if __name__ == '__main__':
    asset_currency  = 'eth'
    endpoint        = 'price'
    exchange_name   = 'binance'
    product_type    = 'linear'
    since           = '2020-05-01'
    # since = '2021-01-01'
    timeframe = '1m'

    kwargs = {key: value for key, value in locals().items() if not key.startswith('__') and isinstance(value, (str, int, float, bool))}


    ##### class 1 #####
    crypto_exchange_data_service = CryptoExchangeDataService(**kwargs)
    # price_df = crypto_exchange_data_service.get_historical_data(True)  # True = underlying asset 的價格數據
    factor_df = crypto_exchange_data_service.get_historical_data()
    sys.exit()

    ##### class 2 #####
    factor_df2 = ConvertCcxtDataSerice.get_historical_data(**kwargs)
    price_df2 = ConvertCcxtDataSerice.get_historical_data(underlying_asset=True, **kwargs)

    ### fetch_mark_ohlcv                  # spot - BTC/USDT, futures = BTC/USDT:USDT  ### fetch_index_ohlcv                 # spot - BTC/USDT, futures = BTC/USDT:USDT  ### fetch_ohlcv                       # spot - BTC/USDT, futures = BTC/USDT:USDT  ### fetch_premium_index_ohlcv         # spot - BTC/USDT, futures = BTC/USDT:USDT  ### fetch_funding_rate_history        # only support linear and inverse  ### fetch_open_interest_history       # only support linear and inverse  ### fetch_long_short_ratio_history    # only support linear and inverse
