import click
from pathlib import Path
import pandas as pd
import pickle
from sklearn.metrics import *
from loguru import logger
import numpy as np


@click.command('collect result main')
@click.option("-i", "--invo-input", "invo_input_files", multiple=True)
@click.option("-t", "--trace-input", "trace_input_files", multiple=True)
@click.option("-o", "--output", "output_file")
def collect_result_main(invo_input_files, trace_input_files, output_file):
    trace_input_files = list(map(lambda _: Path(_), trace_input_files))
    invo_input_files = list(map(lambda _: Path(_), invo_input_files))
    trace_level_trace_ids = set()
    invo_level_trace_ids = set()
    results = []
    for input_file in trace_input_files:
        with open(input_file, 'rb') as f:
            df = pickle.load(f)
        trace_level_trace_ids |= set(df.trace_id)
    for input_file in invo_input_files:
        with open(input_file, 'rb') as f:
            df = pickle.load(f)
        invo_level_trace_ids |= set(df.trace_id)
    trace_ids = trace_level_trace_ids.intersection(invo_level_trace_ids)
    del trace_level_trace_ids, invo_level_trace_ids

    for input_file in trace_input_files:
        with open(input_file, 'rb') as f:
            df = pickle.load(f).set_index(['trace_id'])
        idx = set(df.index.values).intersection(trace_ids)
        y_true = df.loc[idx, 'trace_label'].values
        for algo in ['RF-Trace', 'KNN-Trace', 'MLP-Trace']:
            try:
                y_pred = df.loc[idx, f'{algo}-predict'].values
            except KeyError:
                continue
            for metric_function, metric_name in [
                (f1_score, 'F1-score'), (precision_score, 'Precision'), (recall_score, 'Recall')
            ]:
                results.append({
                    'metric_value': metric_function(y_true, y_pred),
                    'metric_name': metric_name,
                    'tp': np.count_nonzero(y_true & y_pred),
                    'fp': np.count_nonzero((~y_true) & y_pred),
                    'fn': np.count_nonzero(y_true & (~y_pred)),
                    'tn': np.count_nonzero((~y_true) & (~y_pred)),
                    'method': algo,
                    'name': input_file.name.split('.')[0],
                })

    for input_file in invo_input_files:
        with open(input_file, 'rb') as f:
            df = pickle.load(f)
        groupby = df.groupby(by=['trace_id'])
        idx = set(df.trace_id.values).intersection(trace_ids)
        y_true = np.asarray(groupby.first().loc[idx, 'trace_label'].values)
        for method in ["Ours", "NoSelection", "IF"]:
            try:
                y_pred = np.asarray(groupby.sum().loc[idx, f'{method}-predict'].values >= 1)
            except KeyError:
                continue
            for metric_function, metric_name in [
                (f1_score, 'F1-score'), (precision_score, 'Precision'), (recall_score, 'Recall')
            ]:
                results.append({
                    'metric_value': metric_function(y_true, y_pred),
                    'metric_name': metric_name,
                    'tp': np.count_nonzero(y_true & y_pred),
                    'fp': np.count_nonzero((~y_true) & y_pred),
                    'fn': np.count_nonzero(y_true & (~y_pred)),
                    'tn': np.count_nonzero((~y_true) & (~y_pred)),
                    'method': method,
                    'name': input_file.name.split('.')[0],
                })
    results = pd.DataFrame.from_records(results)
    logger.debug(f"results:\n{results}")
    results.to_csv(output_file, index=False)


if __name__ == '__main__':
    collect_result_main()
