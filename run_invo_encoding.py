import sys
from typing import Dict, Tuple
import numpy as np

import click
import pickle
import pandas as pd
from pathlib import Path
from loguru import logger

from data.trainticket.download import simple_name
from trainticket_config import *

"""
Encode train-ticket pickle data into data frame of invocations:
    source, target, start time, end time, trace_id, features
    ...
"""


@click.command('invo-encoding')
@click.option('-i', '--input', 'input_file', default="*.pkl", type=str)
@click.option('-o', '--output', 'output_file', default='', type=str)
# @click.option('-e', '--error-time', default='error_time.pkl', type=str)
def train_ticket_invo_encoding_main(input_file: str, output_file: str):
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)
    # logger.debug(f"input file: {input_file}, output_file: {output_file}")
    with open(str(input_file.resolve()), 'rb') as f:
        input_data = pickle.load(f)

    if ENABLE_ALL_FEATURES:
        data = {
            'source': [], 'target': [], 'start_timestamp': [], 'end_timestamp': [], 'trace_label': [],
            'trace_id': [],
            'latency': [], 'cpu_use': [], 'mem_use_percent': [], 'mem_use_amount': [],
            'file_write_rate': [], 'file_read_rate': [],
            'net_send_rate': [], 'net_receive_rate': [], 'http_status': [],
            'trace_start_timestamp': [], 'trace_end_timestamp': [],
        }
    else:
        data = {
            'source': [], 'target': [], 'start_timestamp': [], 'end_timestamp': [], 'trace_label': [],
            'trace_id': [],
            'latency': [], 'http_status': [],
            'trace_start_timestamp': [], 'trace_end_timestamp': [],
        }

    for trace in input_data:
        indices = np.asarray([idx for idx, (source, target) in enumerate(trace['s_t']) if source != target])
        if len(indices) <= 0:
            continue
        for key, item in trace.items():
            if isinstance(item, list) and key != 'root_cause' and key != 'fault_type':
                try:
                    trace[key] = np.asarray(item)[indices]
                except IndexError:
                    raise RuntimeError(f"{key} {item} {indices}")
        data['source'].extend(list(simple_name(_[0]) for _ in trace['s_t']))
        data['target'].extend(list(simple_name(_[1]) for _ in trace['s_t']))

        if ENABLE_ALL_FEATURES:
            data['start_timestamp'].extend(_ / 1e6 for _ in trace['timestamp'])
            data['end_timestamp'].extend(_ / 1e6 for _ in trace['endtime'])
            data['trace_start_timestamp'].extend(min(trace['timestamp']) / 1e6 for _ in trace['timestamp'])
            data['trace_end_timestamp'].extend(max(trace['endtime']) / 1e6 for _ in trace['endtime'])
            data['trace_label'].extend(trace['label'] for _ in trace['s_t'])
            data['trace_id'].extend(trace['trace_id'] for _ in trace['s_t'])
            data['latency'].extend(_ / 1e6 for _ in trace['latency'])
            data['cpu_use'].extend(_ * 1e-2 for _ in trace['cpu_use'])
            data['mem_use_percent'].extend(_ / 1e2 for _ in trace['mem_use_percent'])  #
            data['mem_use_amount'].extend(_ / 1e12 for _ in trace['mem_use_amount'])  # 1000MB disabled
            data['file_write_rate'].extend(_ / 1e12 for _ in trace['file_write_rate'])  # 100MB
            data['file_read_rate'].extend(_ / 1e12 for _ in trace['file_read_rate'])  # 100MB
            data['net_send_rate'].extend(_ / 1e12 for _ in trace['net_send_rate'])
            data['net_receive_rate'].extend(_ / 1e12 for _ in trace['net_receive_rate'])
            data['http_status'].extend(int(_) // 100 if _ != 0 else 9 for _ in trace['http_status'])
        else:
            data['start_timestamp'].extend(_ for _ in trace['timestamp'])
            data['end_timestamp'].extend(_ for _ in trace['endtime'])
            data['trace_start_timestamp'].extend(min(trace['timestamp']) for _ in trace['timestamp'])
            data['trace_end_timestamp'].extend(max(trace['endtime']) for _ in trace['endtime'])
            data['trace_label'].extend(trace['label'] for _ in trace['s_t'])
            data['trace_id'].extend(trace['trace_id'] for _ in trace['s_t'])
            data['latency'].extend(_ for _ in trace['latency'])
            data['http_status'].extend(int(_) // 100 if _ != 0 else 9 for _ in trace['http_status'])

    df = pd.DataFrame.from_dict(
        data, orient='columns',
    )
    for feature_name in FEATURE_NAMES:
        assert feature_name in df.columns
    for service in np.unique(df.source):
        assert service in INVOLVED_SERVICES, f'{service} {df[df.source == service]}'
    for service in np.unique(df.target):
        assert service in INVOLVED_SERVICES, f'{service} {df[df.source == service]}'
    with open(output_file, 'wb+') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    train_ticket_invo_encoding_main()
