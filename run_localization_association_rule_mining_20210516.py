import time
from collections import defaultdict
import pickle
import random
from collections import defaultdict
from functools import lru_cache, reduce
from pathlib import Path
from typing import List, FrozenSet

import click
import numpy as np
import pandas as pd
from loguru import logger
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from trainticket_config import EXP_NOISE_LIST


PREDICT_COLUMN = 'predict'


def inject_noise(df, ratio=0):
    rng = np.random.default_rng(2021)
    df = df.copy()
    df = df.drop(columns=['source', 'target']).reset_index()
    for idx in range(len(df)):
        if rng.random() < ratio:
            df.loc[idx, PREDICT_COLUMN] = 1 - df.iloc[idx][PREDICT_COLUMN]
    abnormal_traces = set(df.loc[df.predict == 1, 'trace_id'].unique())
    if ratio == 0:
        for idx in range(len(df)):
            if df.iloc[idx]['trace_id'] in abnormal_traces and rng.random() < 0.01:
                df.loc[idx, PREDICT_COLUMN] = 0
    return df.set_index(['source', 'target'], drop=False)


@click.command('association rule mining')
@click.option("-i", "--input", "input_file")
@click.option("-o", "--output", "output_file")
@click.option("--min-support-rate", default=0.1)
@click.option("--enable-PRFL", is_flag=True)
@click.option("--k", default=100)
@click.option("--quiet", "-q", is_flag=True)
def main(input_file, output_file, min_support_rate, quiet, k, enable_prfl):
    input_file = Path(input_file)

    with open(input_file, 'rb+') as f:
        input_file = pickle.load(f)
    # input_file = input_file.reset_index(drop=True).set_index('trace_id')
    # trace_ids = set(input_file.index.unique())
    # input_file = input_file.loc[random.sample(trace_ids, k=int(len(trace_ids) / 1))].reset_index().set_index(
    #     ['source', 'target'], drop=False
    # )

    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)

    output_data = {}
    tracerca = TraceRCA()
    n_traces = len(input_file['trace_id'].unique())
    # for noise in EXP_NOISE_LIST:
    for noise in [0]:
        injected_file = inject_noise(input_file, noise)
        tic = time.time()
        fp = tracerca(
            injected_file,
            min_support_rate=min_support_rate,
            k=k,
            predict_column=PREDICT_COLUMN,
            quiet=quiet,
            forbidden_names=frozenset({"gateway"}),
            enable_PRFL=False
        )
        toc = time.time()
        print(f'{output_file} {noise=} speed={n_traces / (toc - tic):.2f}traces/second, time={toc - tic:.2f}s')
        output_data[f"Ours-noise={noise}"] = fp

    with open(output_file, 'wb+') as f:
        pickle.dump(output_data, f)


class ItemsetHandler:
    def __init__(self, data: pd.DataFrame, **kwargs):
        trace_ids, itemsets = self.gen_itemsets(data, **kwargs)
        self.trace_ids = trace_ids
        self.itemsets = itemsets
        self.items = reduce(lambda a, b: a | b, map(lambda x: x[0], self.itemsets)) \
                     - kwargs.get("forbidden_names", frozenset())

        item2trace_id_sets = defaultdict(set)
        item2abnormal_trace_id_sets = defaultdict(set)
        for trace_id, (node_set, label) in zip(trace_ids, self.itemsets):
            for _item in node_set:
                item2trace_id_sets[_item].add(trace_id)
            if label:
                for _item in node_set:
                    item2abnormal_trace_id_sets[_item].add(trace_id)

        self._traces_containing_item = {key: frozenset(value) for key, value in item2trace_id_sets.items()}
        self._abnormal_traces_containing_item = {
            key: frozenset(value) for key, value in item2abnormal_trace_id_sets.items()
        }

        self.p_b = len(list(filter(lambda x: bool(x[1]), self.itemsets))) / len(self.itemsets)

        self.p_a_given_b, self.frequent_patterns = self._frequent_pattern_on_abnormal(**kwargs)

        self.p_b_given_a = self._get_confidence()

        self.abnormal_itemsets_counts = defaultdict(int)
        for itemset, abnormal in self.itemsets:
            if abnormal:
                self.abnormal_itemsets_counts[itemset] += 1
        self.abnormal_itemset_total_count = sum(self.abnormal_itemsets_counts.values())

    def p_a_given_b_rescaled(self, itemset):
        ret = 0
        cnt = 0
        for key, value in self.abnormal_itemsets_counts.items():
            if key > itemset:
                ret += (value / self.abnormal_itemset_total_count) ** 2
                cnt += 1
        ret *= cnt
        return ret

    def _frequent_pattern_on_abnormal(self, **kwargs):
        min_support_rate = kwargs.get("min_support_rate", 0.1)
        abnormal_itemsets = list(map(lambda x: x[0], filter(lambda x: bool(x[1]), self.itemsets)))
        te = TransactionEncoder()
        te_ary = te.fit_transform(abnormal_itemsets)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        ret = fpgrowth(df, min_support=min_support_rate, use_colnames=True)
        p_a_given_b, frequent_itemsets = ret.support.values, ret.itemsets.values
        return {
            k: v for k, v in zip(frequent_itemsets, p_a_given_b)
        }, frequent_itemsets

    def _get_confidence(self):
        confidence = {}
        for itemset in self.frequent_patterns:
            all_idx = self.traces_containing_pattern(itemset)
            abnormal_idx = self.abnormal_traces_containing_pattern(itemset)
            confidence[itemset] = len(abnormal_idx) / len(all_idx)
        return confidence

    def traces_containing_item(self, item: str):
        return self._traces_containing_item.get(item, frozenset())

    def abnormal_traces_containing_item(self, item: str):
        return self._abnormal_traces_containing_item.get(item, frozenset())

    @lru_cache(maxsize=None)
    def traces_containing_pattern(self, pattern: FrozenSet[str]):
        return reduce(lambda a, b: a & b, map(self.traces_containing_item, pattern))

    @lru_cache(maxsize=None)
    def abnormal_traces_containing_pattern(self, pattern: FrozenSet[str]):
        return reduce(lambda a, b: a | b, map(self.abnormal_traces_containing_item, pattern)) \
               & self.traces_containing_pattern(pattern)

    @staticmethod
    def gen_itemsets(
            data: pd.DataFrame, **kwargs
    ):
        predict_column = kwargs.get("predict_column", "predict")
        source_column = kwargs.get("source_column", "source")
        target_column = kwargs.get("target_column", "target")
        trace_id_column = kwargs.get("trace_id_column", "trace_id")

        forbidden_names = kwargs.get("forbidden_names", frozenset())

        trace_group = data.groupby(trace_id_column)
        trace_predict_ans = trace_group[predict_column].max()
        node_sets = list(
            frozenset(a | b) - forbidden_names
            for a, b in zip(trace_group[target_column].apply(set), trace_group[source_column].apply(set))
        )
        node_sets = list(zip(node_sets, trace_predict_ans))
        return list(map(lambda x: x[0], trace_group)), node_sets


class PRFL:
    def __init__(self, data: pd.DataFrame, itemset_handler: ItemsetHandler, w=0.002, phi=0.5, d=0.6):
        # test data
        # data = pd.read_csv("./PRFL_test.csv")
        # itemset_handler = ItemsetHandler(data)

        tic = time.time()
        self._data = data
        self._ih = itemset_handler

        abnormal_traces = set()
        involved_vertices = set()
        for trace_id, (node_set, label) in zip(self._ih.trace_ids, self._ih.itemsets):
            if label:
                abnormal_traces.add(trace_id)
                involved_vertices |= node_set
        abnormal_traces = list(abnormal_traces)
        involved_vertices = list(involved_vertices)
        vertex2idx = {v: i for i, v in enumerate(involved_vertices)}
        self._vertex2idx = vertex2idx
        trace2idx = {v: i for i, v in enumerate(abnormal_traces)}
        A_oo = np.zeros((len(involved_vertices), len(involved_vertices)), dtype=np.float)
        A_ot = np.zeros((len(involved_vertices), len(abnormal_traces)), dtype=np.float)
        A_to = np.zeros((len(abnormal_traces), len(involved_vertices)), dtype=np.float)

        # traces' kinds
        trace_id_indexed_data = data.set_index(['trace_id'])
        trace2kind = {}
        kind2abnormal_traces = defaultdict(list)
        kind2normal_traces = defaultdict(list)
        for trace_id in data.trace_id.unique():
            kind = frozenset(set(trace_id_indexed_data.loc[trace_id, ['source', 'target']].apply(tuple).unique()))
            trace2kind[trace_id] = kind
            if trace_id in abnormal_traces:
                kind2abnormal_traces[kind].append(trace_id)
            else:
                kind2normal_traces[kind].append(trace_id)

        for (a, b) in data.loc[data.trace_id.isin(abnormal_traces), ['source', 'target']].apply(tuple, axis=1).unique():
            if a in vertex2idx and b in vertex2idx:
                A_oo[vertex2idx[b], vertex2idx[a]] = 1
        A_oo = A_oo / (np.sum(A_oo, axis=0, keepdims=True) + 1e-4)
        for v in involved_vertices:
            for t in self._ih.abnormal_traces_containing_item(v):
                A_ot[vertex2idx[v], trace2idx[t]] = 1
                A_to[trace2idx[t], vertex2idx[v]] = 1
        A_ot = A_ot / (np.sum(A_ot, axis=0, keepdims=True) + 1e-4)
        A_to = A_to / (np.sum(A_to, axis=0, keepdims=True) + 1e-4)
        A = np.block([[A_oo * w, A_ot], [A_to, np.zeros((len(abnormal_traces), len(abnormal_traces)))]])
        # A = A / np.sum(A, axis=1, keepdims=True)

        u_o = np.zeros(len(involved_vertices), dtype=np.float)
        normal_kind = np.array([
            1. / (len(kind2normal_traces[trace2kind[trace_id]]) + 1e-4) for trace_id in abnormal_traces
        ])
        u_t_normal = normal_kind / np.sum(normal_kind)
        abnormal_kind = np.array([
            1. / len(kind2abnormal_traces[trace2kind[trace_id]]) for trace_id in abnormal_traces
        ])
        n_operations = np.zeros_like(abnormal_kind)
        for trace_id, (node_set, label) in zip(self._ih.trace_ids, self._ih.itemsets):
            if trace_id in trace2idx:
                n_operations[trace2idx[trace_id]] = 1 / len(node_set)
        u_t_abnormal = phi * n_operations / np.sum(n_operations) + (1 - phi) * abnormal_kind / np.sum(abnormal_kind)

        v0 = np.block([np.ones_like(u_o) / len(involved_vertices), np.ones_like(u_t_normal) / len(abnormal_traces)])

        self._pi_normal = self.solve_PPR(A, np.block([u_o, u_t_normal]), d=d, v0=v0)[:len(involved_vertices)]
        self._pi_abnormal = self.solve_PPR(A, np.block([u_o, u_t_abnormal]), d=d, v0=v0)[:len(involved_vertices)]
        self._pi_normal /= np.max(self._pi_normal)
        self._pi_abnormal /= np.max(self._pi_abnormal)

        toc = time.time()
        logger.debug(f"PRFL costs {toc - tic:.2f} seconds")
        logger.debug(f"A has shape {np.shape(A)}, and costs {A.nbytes / 1024 / 1024:.2f} MBytes")

    def normal_score(self, name):
        return self._pi_normal[self._vertex2idx[name]]

    def abnormal_score(self, name):
        return self._pi_abnormal[self._vertex2idx[name]]

    @staticmethod
    def solve_PPR(A, u, d, v0):
        rng = np.random.default_rng()
        v = v0.copy()
        v = v / np.sum(v)
        old_v = np.zeros_like(v)
        while np.max(np.abs(v - old_v)) >= 1e-3:
            old_v = v
            v = d * A @ v + (1 - d) * u
        return v


class TraceRCA:
    def __call__(self, input_file: pd.DataFrame, enable_PRFL=False, **kwargs) -> List[str]:
        predict_column = kwargs.get("predict_column", "predict")
        source_column = kwargs.get("source_column", "source")
        target_column = kwargs.get("target_column", "target")
        trace_id_column = kwargs.get("trace_id_column", "trace_id")
        quiet = kwargs.get('quiet', False)
        k = kwargs.get("k", 100)

        input_file[predict_column] = input_file[predict_column].astype("bool")

        itemset_handler = ItemsetHandler(input_file, **kwargs)

        if len(itemset_handler.frequent_patterns) <= 0:
            logger.warning("no frequent pattern")
            return []

        if not enable_PRFL:
            @lru_cache(maxsize=None)
            def jaccard_similarity(_itemset):
                p_a_given_b = itemset_handler.p_a_given_b[_itemset]
                p_b_given_a = itemset_handler.p_b_given_a[_itemset]
                return 1 / (1 / np.maximum(p_a_given_b, 1e-4) + 1 / np.maximum(p_b_given_a, 1e-4) - 1)
        else:
            prfl = PRFL(input_file, itemset_handler)

            @lru_cache(maxsize=None)
            def jaccard_similarity(_itemset):
                normal_score = reduce(
                    lambda x, y: x * y,
                    [prfl.normal_score(_) for _ in _itemset],
                    1
                )
                abnormal_score = reduce(
                    lambda x, y: x * y,
                    [prfl.abnormal_score(_) for _ in _itemset],
                    1
                )
                f = itemset_handler.p_b * len(itemset_handler.itemsets)
                p = len(itemset_handler.itemsets) - f
                # ef = f * abnormal_score
                # nf = f - ef
                # ep = p * normal_score
                # np = p - ep
                ef = abnormal_score * len(itemset_handler.abnormal_traces_containing_pattern(_itemset))
                nf = abnormal_score * f - ef
                ep = normal_score * (
                    len(itemset_handler.traces_containing_pattern(_itemset)) - len(itemset_handler.abnormal_traces_containing_pattern(_itemset))
                )
                np = normal_score * p - ep
                support = ef / f
                confidence = ef / (ef + ep)
                return 1 / (1 / support + 1 / confidence - 1)
        trace_id_indexed = input_file.set_index([trace_id_column])
        trace_id_sets = set(trace_id_indexed.index.values)
        abnormal_data_indexed = input_file.set_index([source_column, target_column, predict_column]).xs(
            True, level=predict_column
        )
        abnormal_target_trace_ids = abnormal_data_indexed.groupby(target_column)[trace_id_column].apply(set).to_dict()
        abnormal_source_trace_ids = abnormal_data_indexed.groupby(source_column)[trace_id_column].apply(set).to_dict()

        def in_out_diff(_item, _pattern, return_="diff") -> float:
            if len(_pattern) <= 0:
                return 0.
            _pattern_trace_ids = itemset_handler.traces_containing_pattern(_pattern) & trace_id_sets
            if len(_pattern_trace_ids) <= 0:
                return 0.
            _target = len(abnormal_target_trace_ids.get(_item, frozenset()) & _pattern_trace_ids)
            _source = len(abnormal_source_trace_ids.get(_item, frozenset()) & _pattern_trace_ids)
            if return_ == "diff":
                # return np.log(np.abs(_target - _source) + 1) + 1
                return np.abs(_target - _source)
            elif return_ == "out":
                return _source
            elif return_ == "in":
                return _target

        pattern_scores = {
            pattern: jaccard_similarity(pattern) for pattern in itemset_handler.frequent_patterns
        }
        pattern_score_threshold = sorted(list(set(pattern_scores.values())), reverse=True)[:k][-1]
        patterns = list(filter(
            lambda x: pattern_scores[x] >= pattern_score_threshold,
            itemset_handler.frequent_patterns
        ))
        patterns = sorted(patterns, key=lambda x: pattern_scores[x], reverse=True)[:1000]
        item_ret = defaultdict(lambda: {"score": -1, "pattern": frozenset(), "pattern_score": 0, "in-pattern_score": 0})
        for pattern in patterns:
            if len(pattern) > 1:
                continue
            for item in pattern:
                # if (ps := pattern_scores[pattern]) > item_ret[item]["score"]:
                ps = pattern_scores[pattern]
                ips = in_out_diff(item, pattern)
                if (s := ips * ps) > item_ret[item]["score"]:
                    item_ret[item] = {
                        "score": s,
                        "pattern": pattern,
                        "pattern_score": ps,
                        "in-pattern_score": ips,
                    }

        ret = sorted(
            itemset_handler.items,
            key=lambda _item: (item_ret[_item]["score"], -len(item_ret[_item]["pattern"])),
            reverse=True,
        )
        if not quiet:
            logger.debug(
                f"|{'item':30}|{'score':8}|{'pattern':60}|{'ps':8}|{'ips':8}|{'p(a|b)':8}|{'p(b|a)':8}|{'in':8}|{'out':8}"
            )
            for item in ret:
                logger.debug(
                    f"|{item:30}|{item_ret[item]['score']:8.2f}"
                    f"|{','.join(item_ret[item]['pattern']):60}"
                    f"|{item_ret[item]['pattern_score']:8.6f}"
                    f"|{item_ret[item]['in-pattern_score']:8.2f}"
                    f"|{itemset_handler.p_a_given_b.get(item_ret[item]['pattern'], 0):8.6f}"
                    f"|{itemset_handler.p_b_given_a.get(item_ret[item]['pattern'], 0):8.6f}"
                    f"|{in_out_diff(item, item_ret[item]['pattern'], return_='in'):8.0f}"
                    f"|{in_out_diff(item, item_ret[item]['pattern'], return_='out'):8.0f}"
                )
        return ret


if __name__ == "__main__":
    main()
