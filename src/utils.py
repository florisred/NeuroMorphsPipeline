from sklearn.preprocessing import StandardScaler


def scale_session(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def calc_mean_per_stimulus(data_df, labels):
    grouped_data = data_df.T.groupby(labels["morph_name"].values).mean()
    grouped_data_labels = labels.drop_duplicates(subset='morph_name').sort_values(by=['morph_name'])
    grouped_data.index = grouped_data_labels['morph_name']
    return grouped_data, grouped_data_labels