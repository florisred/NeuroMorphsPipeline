from src.MixIns.pcaMixin import PCAMixin
from src.MixIns.PlotCreatorMixIn import PlotCreatorMixIn

#ToDo: make sure it requires self.data_df to be present

class AnalysisMixin(PCAMixin, PlotCreatorMixIn,):
    """
    Only requires self.data_df and self.labels to be defined, will handle all analysis
    """

    def create_2d_plots(self, transitions = ''):
        se

        self._create_2d_plots(
            pca=,
            labels=,
            pca_type=,
            plot_name=,

        )
