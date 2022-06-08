import sys
from typing import Dict, Tuple, List
import numpy as np

import click
import pickle
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from data.trainticket.download import simple_name
from trainticket_config import *

"""
Encode train-ticket pickle data into trace-level data and label:
{
'data': array in shape (n_traces, n_features * n_microservices),
'labels': array in shape (n_traces,)
'masks': array in shape (n_traces, n_features * n_microservices),
'trace_ids': array in shape (n_traces)
}
"""


def encoding_data(source_data: List, drop_service=(), drop_fault_type=()):
    def pair2index(s_t):
        return SERVICE2IDX.get(simple_name(s_t[1]))

    if ENABLE_ALL_FEATURES:
        _data = np.ones((len(source_data), len(INVOLVED_SERVICES), 9), dtype=np.float32) * -1
    else:
        _data = np.ones((len(source_data), len(INVOLVED_SERVICES), 2), dtype=np.float32) * -1

    _labels = np.zeros((len(source_data),), dtype=np.bool)
    _trace_ids = [""] * len(source_data)
    _service_mask = np.zeros((len(source_data), len(INVOLVED_SERVICES)), dtype=np.bool)
    _root_causes = np.zeros((len(source_data), len(INVOLVED_SERVICES)), dtype=np.bool)
    for trace_idx, trace in enumerate(source_data):
        if 'fault_type' in trace and trace['fault_type'] in drop_fault_type:
            continue
        if 'root_cause' in trace and any(_ in drop_service for _ in trace['root_cause']):
            continue
        indices = np.asarray([idx for idx, (source, target) in enumerate(trace['s_t']) if source != target])
        if len(indices) <= 0:
            continue
        for key, item in trace.items():
            if isinstance(item, list) and key != 'root_cause' and key != 'fault_type':
                trace[key] = np.asarray(item)[indices]
        service_idx = np.asarray(list(map(pair2index, (trace['s_t']))))
        _service_mask[trace_idx, service_idx] = True
        # assert all(np.diff(trace['endtime']) <= 0), f'end time is not sorted: {trace["endtime"]}'

        if ENABLE_ALL_FEATURES:
            _data[trace_idx, service_idx, 0] = np.asarray(trace['latency']) / 1e6
            _data[trace_idx, service_idx, 1] = np.asarray(trace['cpu_use']) / 100
            _data[trace_idx, service_idx, 2] = np.asarray([round(_, 2) for _ in trace['mem_use_percent']])
            _data[trace_idx, service_idx, 3] = np.asarray(trace['mem_use_amount']) / 1e9  # 1000M
            _data[trace_idx, service_idx, 4] = np.asarray(trace['file_write_rate']) / 1e8
            _data[trace_idx, service_idx, 5] = np.asarray(trace['file_read_rate']) / 1e8
            _data[trace_idx, service_idx, 6] = np.asarray(trace['net_send_rate']) / 1e8
            _data[trace_idx, service_idx, 7] = np.asarray(trace['net_receive_rate']) / 1e8
            _data[trace_idx, service_idx, 8] = list(map(lambda x: x // 100 if x != 0 else 9, (trace['http_status'])))
        else:
            _data[trace_idx, service_idx, 0] = np.asarray(trace['latency'])
            _data[trace_idx, service_idx, 1] = list(map(lambda x: int(x) // 100 if x != 0 else 9, (trace['http_status'])))

        _labels[trace_idx] = trace['label']
        _trace_ids[trace_idx] = trace['trace_id']
        _trace_root_causes = trace['root_cause'] if 'root_cause' in trace else []
        for _root_cause in _trace_root_causes:
            _root_causes[trace_idx, SERVICE2IDX[_root_cause]] = True
    _mask = np.tile(_service_mask[:, :, np.newaxis], (1, 1, 9))
    return _data, _labels, _mask, _trace_ids, _root_causes


@click.command('trace-encoding')
@click.option('-i', '--input', 'input_file', default="*.pkl", type=str)
@click.option('-o', '--output', 'output_file', default='', type=str)
@click.option('--drop-service', default=0)
@click.option('--drop-fault-type', default=0)
def main(*args, **kwargs):
    train_ticket_trace_encoding(*args, **kwargs)


def train_ticket_trace_encoding(input_file: str, output_file: str, drop_service, drop_fault_type):
    drop_service = list(INVOLVED_SERVICES)[:drop_service]
    drop_fault_type = list(FAULT_TYPES)[:drop_fault_type]
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)
    with open(str(input_file.resolve()), 'rb') as f:
        input_data = pickle.load(f)
    data, labels, masks, trace_ids, root_causes = encoding_data(input_data, drop_service, drop_fault_type)
    np.savez(
        output_file,
        data=data.reshape((len(data), -1)),
        labels=labels,
        masks=masks.reshape((len(data), -1)),
        trace_ids=trace_ids,
        root_causes=root_causes,
    )


if __name__ == '__main__':
    main()
