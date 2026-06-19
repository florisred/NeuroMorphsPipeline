import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def classify(datasource_dict: dict, k_folds: int = 5, **kwargs):
    output_dir = kwargs['output_dir'] / 'classification'
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    fractions = [0.1, 0.5, 1.0]
    styles = {
        0.1: (':s', '10% train'),
        0.5: ('--x', '50% train'),
        1.0: ('-o', '100% train')
    }

    for key, data_source in datasource_dict.items():
        test_ds = data_source.copy()
        train_ds = data_source.copy()

        train_x = train_ds.anchors
        train_y = np.array(train_x.index)

        anchor_mask = test_ds.metadata.anchor_mask
        test_x = test_ds.data.loc[~anchor_mask]
        test_y = test_ds.metadata.nearest_anchor.loc[~anchor_mask]
        test_y_norm_step = test_ds.metadata.morph_steps.loc[~anchor_mask]

        for frac in fractions:
            fmt, label_suffix = styles[frac]

            if frac == 1.0:
                classifier = LinearSVC(max_iter=10000)
                classifier.fit(train_x, train_y)
                prediction = classifier.predict(test_x)

                comparison_df = pd.DataFrame({'nearest_anchor': test_y})
                comparison_df['norm_step'] = test_y_norm_step
                comparison_df['prediction'] = prediction
                comparison_df['correct'] = (comparison_df['prediction'] == comparison_df['nearest_anchor'])

                accuracy_per_step = comparison_df.groupby('norm_step')['correct'].mean().sort_index()

                plt.plot(accuracy_per_step.index, accuracy_per_step.values,
                         fmt, label=f'{key} ({label_suffix})', markersize=4, alpha=0.8)

            else:
                n_samples = len(train_x)
                n_classes = len(np.unique(train_y))

                # Force integer sample size. Guarantee at least 'n_classes' samples
                # so the SVM has at least 1 of each class to train on, avoiding 0-sample crashes.
                train_size_int = max(n_classes, int(n_samples * frac))

                # Edge case: Ensure we leave at least 1 sample for the split's test set
                if train_size_int >= n_samples:
                    train_size_int = n_samples - 1

                try:
                    splitter = StratifiedShuffleSplit(n_splits=k_folds, train_size=train_size_int, random_state=42)
                    splits = list(splitter.split(train_x, train_y))
                except ValueError:
                    # Fallback to standard ShuffleSplit if class distributions are too sparse to stratify
                    splitter = ShuffleSplit(n_splits=k_folds, train_size=train_size_int, random_state=42)
                    splits = list(splitter.split(train_x, train_y))

                fold_accuracies = []

                for train_idx, _ in splits:
                    fold_train_x = train_x.iloc[train_idx] if hasattr(train_x, 'iloc') else train_x[train_idx]
                    fold_train_y = train_y[train_idx]

                    classifier = LinearSVC(max_iter=10000)
                    classifier.fit(fold_train_x, fold_train_y)
                    prediction = classifier.predict(test_x)

                    comparison_df = pd.DataFrame({'nearest_anchor': test_y})
                    comparison_df['norm_step'] = test_y_norm_step
                    comparison_df['prediction'] = prediction
                    comparison_df['correct'] = (comparison_df['prediction'] == comparison_df['nearest_anchor'])

                    acc_step = comparison_df.groupby('norm_step')['correct'].mean().sort_index()
                    fold_accuracies.append(acc_step)

                avg_accuracy_per_step = pd.concat(fold_accuracies, axis=1).mean(axis=1)

                plt.plot(avg_accuracy_per_step.index, avg_accuracy_per_step.values,
                         fmt, label=f'{key} ({label_suffix}, avg {k_folds} folds)', markersize=4, alpha=0.8)

    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Morph Progress (Norm Step)')
    plt.ylabel('Classification Accuracy')
    plt.title('V1 Texture Discrimination across Morphs')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/classification.svg')