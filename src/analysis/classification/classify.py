import pandas as pd

from src.data_objects.data_source import DataSource
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt


def classify(datasource_dict: dict[str, DataSource], **kwargs):
    plt.figure(figsize=(10, 6))

    for key, data_source in datasource_dict.items():
        test_ds = data_source.copy()
        train_ds = data_source.copy()

        train_x = train_ds.anchors
        train_y = np.array(train_x.index)

        anchor_mask = test_ds.metadata.anchor_mask
        test_x = test_ds.data.loc[~anchor_mask]
        test_y = test_ds.metadata.nearest_anchor.loc[~anchor_mask]
        test_y_norm_step = test_ds.metadata.morph_steps.loc[~anchor_mask]

        classifier = LinearSVC()
        classifier.fit(train_x, train_y)
        prediction = classifier.predict(test_x)

        comparison_df = pd.DataFrame(test_y)
        comparison_df['norm_step'] = test_y_norm_step
        comparison_df['prediction'] = prediction
        comparison_df['correct'] = (comparison_df['prediction'] == comparison_df['nearest_anchor'])

        accuracy_per_step = comparison_df.groupby('norm_step')['correct'].mean().sort_index()

        plt.plot(accuracy_per_step.index, accuracy_per_step.values, '-o', label=key, markersize=4, alpha=0.8)

    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Morph Progress (Norm Step)')
    plt.ylabel('Classification Accuracy')
    plt.title('V1 Texture Discrimination across Morphs')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

