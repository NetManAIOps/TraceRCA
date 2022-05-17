import pickle

import click
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler
from trainticket_config import FEATURE_NAMES


def extract_data(path):
    x = np.load(path)
    return x['data'], x['labels'], x['masks'], x['trace_ids']


@click.command('train-anoamly-detection-model')
@click.option('-i', '--invo-history', default='historical_data.pkl', type=str)
@click.option('-t', '--trace-history', default='historical_data.pkl', type=str)
@click.option('-o', '--output-file', type=str)
def main(trace_history, invo_history, output_file):
    his_data, his_labels, his_masks, his_trace_ids = extract_data(trace_history)
    # rus = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = rus.fit_resample(his_data, his_labels)

    result = {}
    for algorithm in ['RF-Trace', 'MLP-Trace']:
        if algorithm == 'RF-Trace':
            model = RandomForestClassifier(n_estimators=100, n_jobs=10, verbose=0)
        elif algorithm == 'MLP-Trace':
            model = MLPClassifier(batch_size=256, early_stopping=True, verbose=0, learning_rate_init=1e-4,
                                  max_iter=100, hidden_layer_sizes=(100, 100))
        elif algorithm == 'KNN-Trace':
            model = KNeighborsClassifier()
        else:
            raise RuntimeError()
        model.fit(his_data, his_labels)
        result[algorithm] = model

    with open(invo_history, 'rb') as f:
        invo_history = pickle.load(f)
    invo_history = invo_history.set_index(keys=['source', 'target'], drop=False).sort_index()
    indices = np.unique(invo_history.index.values)
    for source, target in indices:
        reference = invo_history.loc[(source, target), FEATURE_NAMES].values
        token = f"IF-{source}-{target}"
        model = IsolationForest(behaviour='new', contamination=0.01, n_jobs=10)
        model.fit(reference)
        result[token] = model

    for source, target in indices:
        for feature in FEATURE_NAMES:
            reference = invo_history.loc[(source, target), feature].values
            token = f"reference-{source}-{target}-{feature}-mean-variance"
            result[token] = {
                'mean': np.mean(reference[:]),
                'std': np.maximum(np.std(reference[:]), 0.1)
            }

    with open(output_file, 'wb+') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()
