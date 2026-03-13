from MixIns.PlotMixIns import Plot2DMixIn, PlotInteractiveMixIn, PlotDistancesMixIn

class PlotCreator(Plot2DMixIn, PlotInteractiveMixIn, PlotDistancesMixIn):

    def __init__(self):
        self._pca_data = None


