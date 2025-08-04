# Standard library imports


# Third-party imports

# Custom imports
from analysis_tools.crypto_data_service import CryptoDataService
from analysis_tools.generate_image import EquityCurveGenerator, HeatmapGenerator
from analysis_tools.utilities import Utilities
from overfitting_prevention.cross_validator import CrossValidator
from overfitting_prevention.parameter_plateau import ParameterPlateau
from overfitting_prevention.walk_forward_analysis import WalkForwardAnalysis


class DataAnalysis:

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.backtest_mode = kwargs['backtest_mode']
        self.cross_validate = kwargs['cross_validate']
        self.dash_board = kwargs['dash_board']
        self.generate_equity_curve = kwargs['generate_equity_curve']
        self.parameter_plateau = kwargs['parameter_plateau']
        self.walk_forward = kwargs['walk_forward']


    def data_analysis(self,):
        all_results = {}
        if self.backtest_mode:
            all_results = CryptoDataService(self.kwargs).generate_all_backtest_results()

        Utilities.simple_filtering(self.kwargs)

        # if self.dash_board:
        #     DashBoardGenerator(self.kwargs['asset_currency'], self.kwargs).function_a()


        if self.walk_forward:
            WalkForwardAnalysis(self.kwargs['asset_currency'], self.kwargs).run_walk_forward_analysis()
        if self.parameter_plateau:
            ParameterPlateau(self.kwargs['asset_currency'], self.kwargs).optimize_parameters()
        if self.cross_validate:
            CrossValidator(self.kwargs['asset_currency'], self.kwargs).process_backtest_results()

        HeatmapGenerator(self.kwargs['asset_currency'], self.kwargs).generate_all_heatmaps(all_results, self.kwargs['show_heatmap'], self.kwargs['target_metric'])
        if self.generate_equity_curve:
            EquityCurveGenerator(self.kwargs['asset_currency'], self.kwargs).generate_equity_curves(all_results, self.kwargs['show_equity_curve'])