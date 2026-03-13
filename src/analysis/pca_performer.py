import pandas as pd
from sklearn.decomposition import PCA
from data_objects.TrialMetadata import TrialMetadata
from data_objects.PCAData import PCAData


class PCAPerformer:

    @staticmethod
    def run_pca(pca_type: str, metadata: TrialMetadata, all_data: pd.DataFrame, n_components: int, fit_data: pd.DataFrame=None):
        pca_model = PCA(n_components=n_components)
        if fit_data is not None:
            pca_model.fit(fit_data)
            pca_result = pca_model.transform(all_data)
        else:
            pca_result = pca_model.fit_transform(all_data)
        pca_data = PCAData(
            pca_output = pca_result,
            metadata = metadata,
            pca_type = pca_type
        )
        return pca_data


