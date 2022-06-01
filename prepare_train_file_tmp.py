# prepare train file before more injected data is got
import random
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    train_dir = Path('train')
    test_dir = Path('test')
    for test_file in tqdm(list(test_dir.glob("*.pkl"))):
        with open(str(test_file), 'rb') as f:
            test_data = pickle.load(f)
        train_length = int(len(test_data) * 0.2)
        with open(str(train_dir / test_file.name), 'wb+') as f:
            pickle.dump(test_data[:train_length], f)
        with open(str(test_file), 'wb+') as f:
            pickle.dump(test_data[train_length:], f)


if __name__ == '__main__':
    main()
