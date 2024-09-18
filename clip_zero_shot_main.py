import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
import torch.optim as optim

from configs.clip_config import config
from utils.data_related import data_split, get_dataloader
from transforms.clip_transform import ClipProcessor
from dataset.clip_dataset import ClipCustomDataset
from models.clip_model import ClipCustomModel
from losses.clip_loss import CLIPLoss
from trainers.clip_trainer import CLIPTrainer
from utils.inference import inference_clip, load_model

def test():
    test_info = pd.read_csv(config.test_data_info_file_path)
    label_info = pd.read_csv(config.train_data_info_file_path)

    test_transform = ClipProcessor(config.model_name)

    train_dataset = ClipCustomDataset(config.train_data_dir_path,
                                  label_info,
                                  test_transform,
                                  is_inference = False)

    global label_to_text
    label_to_text = {k: torch.tensor(v, device=config.device) for k, v in train_dataset.label_to_text_res.items()}

    del train_dataset, label_info

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
    
    model = ClipCustomModel(config.model_name)

    predictions = inference_clip(model, 
                            config.device, 
                            test_loader,
                            label_to_text)
    
    print(predictions)
    print(len(predictions))

    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("clip_zero_shot_output.csv", index=False)

if __name__ == "__main__":
    # main()
    test()