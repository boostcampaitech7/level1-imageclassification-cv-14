import os
import pandas as pd

from configs.base_config import config
from utils.inference import csv_soft_voting

def main():
    test_info = pd.read_csv(config.test_data_info_file_path)

    input_paths = os.listdir("./ensemble_probs")

    preds = csv_soft_voting(input_paths, config.num_classes)

    test_info['target'] = preds
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(config.output_name, index=False)

if __name__ == "__main__":
    main()