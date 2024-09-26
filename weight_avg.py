import pandas as pd
import torch.optim as optim


from configs.base_config import config
from utils.data_related import data_split, get_dataloader
from dataset.dataset import CustomDataset
from models.ViT import ViTModel
from utils.inference import inference_vit,extract_probs, save_probs
from transforms.vit_transform import ViTAutoImageTransform
from utils.model_read import get_model, model_load

def main():
    state_dicts = model_load(config.device)
    model = ViTModel('google/vit-large-patch16-384', config.num_classes)
    alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
    model = get_model(state_dicts, alphal, model)
    model.to(config.device)

    test_info = pd.read_csv(config.test_data_info_file_path)

    test_transform = ViTAutoImageTransform()

    test_dataset = CustomDataset(config.test_data_dir_path,
                                 test_info,
                                 test_transform,
                                 is_inference=True)

    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=config.test_shuffle,
                                 num_workers = config.num_workers,
                                 drop_last=False)
    
    
    predictions = extract_probs([model],
                                test_loader,
                                config.device,
                                config.num_classes,
                                inference_vit,
                                )
    test_info = save_probs(test_info, predictions)
    # test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("ViT-L_Weight_Avg_Probs.csv", index=False)






if __name__ == "__main__":
    main()