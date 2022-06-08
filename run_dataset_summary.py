import click
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import re
from datetime import datetime


def extract_data(path):
    x = np.load(path)
    return x['data'], x['labels'], x['masks'], x['trace_ids'], x['root_causes']


@click.command()
@click.option("-i", "--input", "input_files", multiple=True)
def main(input_files):
    invo_data_list = []
    for input_file in tqdm(input_files):
        if input_file.endswith('.npz'):
            pass
        elif input_file.endswith('invo.pkl'):
            with open(input_file, 'rb') as f:
                invo_data_list.append(pickle.load(f))
        else:
            logger.warning(f"not recongnized file: {input_file}")
    invo_data = pd.concat(invo_data_list, ignore_index=True)

    # numbers
    logger.info(f"# Traces: {len(invo_data.trace_id.unique())}")
    logger.info(f"# Invocations: {len(invo_data)}")

    logger.info(f"# Abnormal Traces: {len(invo_data[invo_data.trace_label == 1].trace_id.unique())}")
    logger.info(f"# Affected Invocations: {len(invo_data[invo_data.trace_label == 1])}")
    logger.info(f"# Injections: "
                f"{len(list(filter(lambda filename: 'normal' not in filename and filename.endswith('.invo.pkl'), input_files)))}")

    logger.info(f"time range: {datetime.fromtimestamp(invo_data.start_timestamp.min())} "
                f"{datetime.fromtimestamp(invo_data.end_timestamp.max())}")


if __name__ == '__main__':
    main()
