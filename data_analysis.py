# Standard library imports


# Third-party imports
import pandas as pd

# Custom imports
from cross_validator import CrossValidator
from crypto_data_service import CryptoDataService
from utilities import Utilities






class DataAnalysis:



    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.cross_validate = kwargs['cross_validate']
        self.minimum_sharpe = kwargs['minimum_sharpe']


    def data_analysis(self,):
        all_results = CryptoDataService(self.kwargs).generate_all_backtest_results()

        Utilities.generate_heatmap(all_results, True)

        if self.cross_validate is True and all_results:
            backtest_dataframe_keys, filtered_results, result_data_list = Utilities.filter_results_by_sharpe_ratio(all_results, self.minimum_sharpe)
            cross_validation_result = CrossValidator.generate_tasks_from_results(backtest_dataframe_keys, filtered_results, self.minimum_sharpe, self.kwargs)
            merged_df = pd.merge(pd.DataFrame(result_data_list).sort_values(by='sharpe', ascending=False).reset_index(drop=True), pd.DataFrame(cross_validation_result), on='strategy', how='inner')
            cols = [col for col in merged_df.columns if col != 'strategy'] + ['strategy']
            merged_df = merged_df[cols]
            pass



