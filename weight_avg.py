import torch
import pandas as pd
import numpy as np


from configs.efficientnet_config import config
from utils.data_related import get_dataloader
from transforms.efficientnet_transform import EfficientNetTransform
from dataset.dataset import CustomDataset
from models.base_timm_model import TimmModel
from utils.inference import inference, save_probs
from transforms.clip_transform import ClipProcessor
from utils.model_soup import get_model, model_load

def main():
    model = TimmModel(config.model_name,
                      config.num_classes,
                      False)

    state_dicts = model_load(config.device, config.save_result_path)
    model = get_model(state_dicts, 1 / len(state_dicts), model)

    model.to(config.device)

    test_info = pd.read_csv(config.test_data_info_file_path)

    test_transform = EfficientNetTransform(is_train=False)

    test_dataset = CustomDataset(config.test_data_dir_path,
                                  test_info,
                                  test_transform,
                                  is_inference=True)
    
    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 shuffle=config.test_shuffle,
                                 drop_last=False)
    
    
    predictions = inference(model, 
                                config.device, 
                                test_loader)
    
    predictions = np.array(predictions)

    # test_info['target'] = predictions
    test_info = save_probs(test_info, predictions)
    # test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(config.output_name, index=False)

if __name__ == "__main__":
    main()