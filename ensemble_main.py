import os
import pandas as pd

from configs.base_config import config
from utils.inference import csv_soft_voting, csv_hard_voting, csv_weighted_voting

def main():
    test_info = pd.read_csv(config.test_data_info_file_path)

    input_paths = [os.path.join(config.ensemble_folder, path) for path in os.listdir(config.ensemble_folder)].sort()
    preds = csv_weighted_voting(input_paths, config.num_classes)

    test_info['target'] = preds
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(config.ensemble_output, index=False)

if __name__ == "__main__":
    main()