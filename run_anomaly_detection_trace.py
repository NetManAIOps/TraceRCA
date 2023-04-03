import pickle
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from diskcache import Cache

DEBUG = True


def extract_data(path):
    x = np.load(path)
    return x['data'], x['labels'], x['masks'], x['trace_ids']


def anomaly_detection(train_data, train_labels, test_data, model_cache: Cache, algorithm='RF'):
    model = model_cache.get(algorithm)
    predict = model.predict(test_data)
    assert tuple(predict.shape) == (len(test_data),), f'wrong shape: {predict.shape}'
    return predict


@click.command('invocation anomaly detection')
@click.option('-i', '--input', 'input_file', default="*.pkl", type=str)
@click.option('-o', '--output', 'output_file', default='.', type=str)
@click.option('-m', '--model', 'model_file', default='.', type=str)
@click.option('-h', '--history', default='historical_data.pkl', type=str)
def trace_anomaly_detection_main(input_file, output_file, history, model_file):
    with open(model_file, 'rb+') as f:
        cache = pickle.load(f)

    data, labels, masks, trace_ids = extract_data(input_file)
    his_data, his_labels, his_masks, his_trace_ids = extract_data(history)
    idx = np.concatenate([
        np.where(his_labels == 1)[0],
        np.random.choice(np.where(his_labels == 0)[0], np.count_nonzero(his_labels == 1) * 10)]
    )
    his_data = his_data[idx]
    his_labels = his_labels[idx]
    his_masks = his_masks[idx]
    his_trace_ids = his_trace_ids[idx]

    tic = time.time()
    rf_result = anomaly_detection(his_data, his_labels, data, model_cache=cache, algorithm='RF-Trace')
    toc = time.time()
    print("algo:", "RF", "time:", toc - tic, 'invos:', len(labels))
    tic = time.time()
    mlp_result = anomaly_detection(his_data, his_labels, data, model_cache=cache, algorithm='MLP-Trace')
    toc = time.time()
    print("algo:", "MLP", "time:", toc - tic, 'invos:', len(labels))
    # knn_result = anomaly_detection(train_data, train_labels, test_data, algorithm='KNN')

    df = pd.DataFrame()
    df['trace_id'] = trace_ids
    df['RF-Trace-predict'] = rf_result
    df['MLP-Trace-predict'] = mlp_result
    # df['KNN-predict'] = knn_result
    df['trace_label'] = labels
    with open(output_file, 'wb+') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    trace_anomaly_detection_main()

