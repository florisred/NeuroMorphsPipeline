from pathlib import Path
class AnalysisHelperMixin:

    def _validate_request(self, **kwargs):

        output_dir = kwargs.get('output_dir')
        if not output_dir or type(output_dir) is not Path:
            raise ValueError('output_dir cannot be None and must be a Path object')
        plot_types = kwargs.get('plot_types')
        if not plot_types or type(plot_types) is not list:
            raise ValueError('plot_types cannot be None and must be a list')
        for plot_type in plot_types:
            if plot_type not in self._possible_plot_types:
                raise ValueError(f'plot_type {plot_type} is not a valid plot_type')


