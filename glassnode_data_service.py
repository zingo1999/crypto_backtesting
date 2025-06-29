# Standard library imports
import os
import time


# Third-party imports
import pandas as pd
import requests


# Custom imports
from config import Credentials


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class GlassnodeDataService:

    DATA_SOURCE = 'glassnode'
    DATA_FOLDER = f"data/{DATA_SOURCE}"
    DATA_SOURCE_KEY = Credentials(DATA_SOURCE).api_key
    os.makedirs(DATA_FOLDER, exist_ok=True)

    file_formats = ['pkl', 'parquet', 'csv']
    for fmt in file_formats:
        file_path = f"{DATA_FOLDER}/metadata.{fmt}"
        if os.path.isfile(file_path):
            load_func = {
                'pkl': pd.read_pickle,
                'parquet': pd.read_parquet,
                'csv': pd.read_csv}[fmt]
            metadata_df = load_func(file_path)
            break

    response = requests.get('https://api.glassnode.com/v1/metadata/assets', params={'api_key': DATA_SOURCE_KEY})
    symbol_list = list({asset['symbol'] for asset in response.json()['data']})

    response = requests.get('https://api.glassnode.com/v1/metadata/metrics', params={'api_key': DATA_SOURCE_KEY})
    metadata_list = response.json()

    def __init__(self, parameters):
        self.base_url = "https://api.glassnode.com/v1/metrics"
        self.factor_currency = parameters['factor_currency']
        self.endpoint = parameters['endpoint']
        self.since = parameters['since']
        self.timeframe = parameters['timeframe']


    def get_endpoint_list(self):
        endpoint_list = [item for item in self.metadata_list if self.endpoint in item]
        return endpoint_list

    def fetch_data(self, exchange_name='deribit'):

        # factor_currency = self.factor_currency.upper()
        try:
            metadata_df = self.metadata_df[self.metadata_df['path'].str.contains(self.endpoint)].reset_index(drop=True) if isinstance(self.endpoint, str) else self.metadata_df
        except:
            metadata_df = pd.DataFrame([self.endpoint], columns=['path'])
        since = self.since if not isinstance(self.since, str) else int(pd.to_datetime(self.since).timestamp())
        if since > 1_00_000_000_000: int(since / 1000)
        timeframe = self.timeframe if self.timeframe in ['10m', '1h', '24h'] else '1h'


        endpoint_path = metadata_df['path'].iloc[0]
        category = endpoint_path.split('/')[-2]
        endpoint = endpoint_path.split('/')[-1]

        print(endpoint)

        params = {
                'api_key': self.DATA_SOURCE_KEY,
                'a': self.factor_currency,
                'e': exchange_name,
                'i': timeframe,
                's': since
}
        try:
            response = requests.get(f"{self.base_url}{endpoint_path}", params=params)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                return df
            else:
                print(f"{self.factor_currency} <<{endpoint}>>  - {response.text}.")
                return pd.DataFrame()
        except Exception as e:
            raise e

        # for metric in self.metadata_list:
        #     for factor_currency in ['BTC', 'ETH', 'SOL']:
        #         params = {
        #         'api_key': self.DATA_SOURCE_KEY,
        #         'a': factor_currency,
        #         'e': 'deribit',
        #         'i': timeframe,
        #         's': self.since}
        #
        #         try:
        #             response = requests.get(f"{self.base_url}{metric}", params=params)
        #             if response.status_code == 200:
        #                 data = response.json()
        #                 df = pd.DataFrame(data)
        #                 return df
        #             else:
        #                 print(factor_currency, metric, response.text)
        #         except Exception as e:
        #             raise e
        #         time.sleep(1)






if __name__ == '__main__':
    factor_currency = 'BTC'
    since = int(pd.to_datetime('2020-05-05').timestamp())
    timeframe = '1h'
    endpoint = 'options_25delta_skew_3_months'

    parameters = {key: value for key, value in locals().items() if not key.startswith('__') and isinstance(value, (str, int, float, bool))}

    gn = GlassnodeDataService(parameters)

    data = gn.fetch_data()

    key = gn.DATA_SOURCE_KEY
    response = requests.get('https://api.glassnode.com/v1/metadata/metrics', params={'api_key': key})
    meta_data_list = response.json()


    response = requests.get("https://api.glassnode.com/v1/metadata/assets?filter=asset.blockchains.exists(b,b.on_chain_support==true)", params={'api_key': key})
    data = response.json()
    pass

