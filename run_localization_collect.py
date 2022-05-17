import pickle
from itertools import product
from typing import List, Set
from pathlib import Path
import click
import pandas as pd
from loguru import logger
import ast
from tqdm import tqdm
from numba import jit

from data.trainticket.download import simple_name
from trainticket_config import INVOLVED_SERVICES

FAULT_TYPES = {'delay', 'abort', 'cpu'}


def root_cause_intersection(y_true, y_pred) -> int:
    cnt = 0
    # y_true = [y_true]
    for item_a in y_true:
        for item_b in y_pred:
            if tuple(item_b[:len(item_a)]) == tuple(item_a):
                cnt += 1
                break
    return cnt


def top_k_precision(y_true: Set, y_pred: List, k=1):
    return root_cause_intersection(y_true, y_pred[:k]) / k


def top_k_recall(y_true: Set, y_pred: List, k=1):
    return root_cause_intersection(y_true, y_pred[:k]) / len(y_true)


def get_rank(item_a, y_pred: List):
    for idx, item_b in enumerate(y_pred):
        if tuple(item_b[:len(item_a)]) == tuple(item_a):
            return idx + 1
    else:
        return len(y_pred) + 1


def MFR(y_true: Set, y_pred: List):
    return min(list(map(lambda item_a: get_rank(item_a, y_pred), y_true)))


def MAR(y_true: Set, y_pred: List):
    return sum(list(map(lambda item_a: get_rank(item_a, y_pred), y_true))) / len(y_true)


@click.command("")
@click.option("-i", "--input_file", "input_file_list", multiple=True)
@click.option("-o", "--output", "output_file")
@click.option("-r", "--root-cause", default="")
def main(input_file_list: List, output_file: str, root_cause):
    root_cause = Path(root_cause) if root_cause != "" else None
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)

    input_file_list = list(map(lambda x: Path(x), input_file_list))
    result_records = []
    rf_not_work = set()
    ours_not_work = set()
    for input_file in tqdm(input_file_list):
        filename = input_file.name.split(".")[0]
        attrs = filename.split('_')
        if root_cause is None:
            # for idx, attr in enumerate(attrs):
            #     if attr in FAULT_TYPES:
            #         fault_type = attr
            #         ground_truth = set(frozenset([f"ts-{_}-service"]) for _ in attrs[idx - 1].split('+'))
            #         break
            # else:
            #     raise RuntimeError(f"No fault type found: {input_file}")
            pass
        else:
            with open(str(root_cause / f'{filename}.pkl'), 'rb') as f:
                ground_truth = pickle.load(f)
            if isinstance(ground_truth, str):
                ground_truth = [[simple_name(ground_truth)]]
            elif isinstance(ground_truth, (list, set, tuple)) and isinstance(list(ground_truth)[0], (set, list, tuple)):
                ground_truth = list(list(map(simple_name, _)) for _ in ground_truth)
            elif isinstance(ground_truth, (list, set, tuple)):
                ground_truth = list(map(lambda x: [simple_name(x)], ground_truth))
            else:
                raise RuntimeError(f"ground truth: {ground_truth}")
        # if len(ground_truth) <= 1:
        #     continue
        with open(input_file, 'rb') as f:
            result = pickle.load(f)
        for method, method_data in result.items():
            root_causes = method_data
            # print(ground_truth, root_causes, method)
            if len(root_causes) <= 0:
                root_causes = list()
            elif isinstance(root_causes[0], str):
                root_causes = list([[_] for _ in root_causes])
            elif isinstance(root_causes[0], frozenset):
                root_causes = list(list(_) for _ in root_causes)
            result_records.append({
                'filename': filename,
                'method': method,
                'metric_value': MAR(ground_truth, root_causes),
                'metric_name': f'MAR'
            })
            result_records.append({
                'filename': filename,
                'method': method,
                'metric_value': MFR(ground_truth, root_causes),
                'metric_name': f'MFR'
            })
            for k in (1, 2, 3):
                result_records.append({
                    'filename': filename,
                    'method': method,
                    'metric_value': top_k_recall(ground_truth, root_causes, k=k),
                    'metric_name': f'Top-{k} Accuracy'
                })
                if k == 1 and method == 'Ours-noise=0' and top_k_recall(ground_truth, root_causes, k=k) < 0.5:
                    print(f"========================================\n"
                          f"file: {filename:50} \n"
                          f"ground_truth: {ground_truth!r:100} \n"
                          f"root causes: {root_causes[:3]!s:100}")
                if method == 'RF' and top_k_recall(ground_truth, root_causes, k=10) < 1:
                    rf_not_work.add(filename)
                if method == 'Ours-noise=0' and top_k_recall(ground_truth, root_causes, k=10) < 1:
                    ours_not_work.add(filename)
    result_df = pd.DataFrame.from_records(result_records)
    pd.set_option('display.max_rows', 5000)
    for metric, method in product(('Top-1 Accuracy', 'Top-2 Accuracy', 'Top-3 Accuracy', 'MAR', 'MFR'), ('Ours', 'RandomWalk', 'RCSF', 'microscope')):
        selector = (result_df.method == f'{method}-noise=0') & (result_df.metric_name == f'{metric}')
        logger.debug(
            f"{method} {metric}: "
            f"{result_df[selector].metric_value.mean()}"
        )
    for metric, method in product(('Top-1 Accuracy', 'Top-2 Accuracy', 'Top-3 Accuracy', 'MAR', 'MFR'), ('RF', 'MLP')):
        selector = (result_df.method == f'{method}') & (result_df.metric_name == f'{metric}')
        logger.debug(
            f"{method} {metric}: "
            f"{result_df[selector].metric_value.mean()}")
    result_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
