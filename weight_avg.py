import torch
import pandas as pd
import numpy as np


from configs.clip_config import config
from utils.data_related import get_dataloader
from dataset.clip_dataset import ClipCustomDataset
from models.clip_model import ClipCustomModel
from utils.inference import inference_clip, save_probs
from transforms.clip_transform import ClipProcessor
from utils.model_soup import get_model, model_load

def main():
    data_info = pd.read_csv(config.train_data_info_file_path)

    total_transform = ClipProcessor(transform_name=config.transform_name)

    total_dataset = ClipCustomDataset(config.train_data_dir_path,
                                  data_info,
                                  total_transform,
                                  is_inference = False)
    
    label_to_text = {k: torch.tensor(v, device=config.device) for k, v in total_dataset.label_to_text_res.items()}

    model = ClipCustomModel(config.model_name)

    state_dicts = model_load(config.device, config.save_result_path)
    model = get_model(state_dicts, 1 / len(state_dicts), model)

    model.to(config.device)

    test_info = pd.read_csv(config.test_data_info_file_path)

    test_transform = ClipProcessor(config.model_name)

    test_dataset = ClipCustomDataset(config.test_data_dir_path,
                                  test_info,
                                  test_transform,
                                  is_inference=True)
    
    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 shuffle=config.test_shuffle,
                                 drop_last=False,
                                 collate_fn=test_dataset.preprocess)
    
    
    predictions = inference_clip(model, 
                                config.device, 
                                test_loader,
                                label_to_text)
    
    predictions = np.array(predictions)

    # test_info['target'] = predictions
    test_info = save_probs(test_info, predictions)
    # test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(config.output_name, index=False)

if __name__ == "__main__":
    main()