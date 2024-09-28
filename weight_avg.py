import pandas as pd
import torch.optim as optim


from configs.yj_cv_config import config
from utils.data_related import data_split, get_dataloader
from dataset.dataset import CustomDataset
from models.yj_convnext_model import Convnext_Model
from utils.inference import inference, load_model , inference_convnext
from transforms.yj_cv_transform import AlbumentationsTransform
from utils.Model_Soup import get_model, model_load

def main():
    state_dicts = model_load(config.device)
    model =  Convnext_Model(model_name = 'convnext_xxlarge.clip_laion2b_soup_ft_in1k', num_classes = 500, pretrained = True)
    alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
    model = get_model(state_dicts, alphal, model)
    model.to(config.device)

    test_info = pd.read_csv(config.test_data_info_file_path)

    test_transform = AlbumentationsTransform(is_train=False)


    test_dataset = CustomDataset(config.test_data_dir_path,
                                 test_info,
                                 test_transform,
                                 is_inference=True)

    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=config.test_shuffle,
                                 num_workers = config.num_workers,
                                 drop_last=False)
    
    
    predictions = inference_convnext(model,
                            config.device,
                            test_loader)

    test_info['target'] = predictions
    # test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("probs_convnext_xxlarge.csv", index=False)






if __name__ == "__main__":
    main()