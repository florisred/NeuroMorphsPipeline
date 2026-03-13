from MixIns.plot_mixins import Plot2DMixIn, PlotInteractiveMixIn, PlotDistancesMixIn
from MixIns.rdm_mixin import RDMMixIn

class PlotCreator(Plot2DMixIn, PlotInteractiveMixIn, PlotDistancesMixIn, RDMMixIn):

    def __init__(self):
        self._pca_data = None


