# Standard library imports


# Third-party imports
import pandas as pd

# Custom imports
from cross_validator import CrossValidator
from crypto_data_service import CryptoDataService
from utilities import Utilities
from walk_forward_analysis import WalkForwardAnalysis






class DataAnalysis:



    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.backtest_mode = kwargs['backtest_mode']
        self.cross_validate = kwargs['cross_validate']
        self.minimum_sharpe = kwargs['minimum_sharpe']
        self.walk_forward = kwargs['walk_forward']


    def data_analysis(self,):
        all_results = {}
        if self.backtest_mode:
            all_results = CryptoDataService(self.kwargs).generate_all_backtest_results()

        Utilities.simple_filtering(self.kwargs)

        if self.walk_forward:
            WalkForwardAnalysis(self.kwargs['asset_currency'], self.kwargs).run_walk_forward_analysis()

        Utilities.generate_heatmap(all_results, self.kwargs['show_heatmap'])

        if self.cross_validate:
            CrossValidator(self.kwargs['asset_currency'], self.kwargs).process_backtest_results()
