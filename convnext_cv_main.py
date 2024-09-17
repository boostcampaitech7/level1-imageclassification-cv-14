import pandas as pd
import torch.optim as optim

from configs.base_config import config
from utils.data_related import data_split, get_dataloader
from transforms.convnext_transform import ConvnextTransform
from dataset.dataset import CustomDataset
from models.convnext_model import Convnext_Model
from losses.cross_entropy_loss import CrossEntropyLoss
from trainers.cv_trainer import Trainer
from utils.inference import inference, load_model, inference_convnext
from losses.Focal_Loss import FocalLoss


def main():
    train_info = pd.read_csv(config.train_data_info_file_path)

    train_df, val_df = data_split(train_info, config.test_size, train_info['target'])

    train_transform = ConvnextTransform(is_train=True)

    train_dataset = CustomDataset(config.train_data_dir_path,
                                  train_df,
                                  train_transform)
    
    model = Convnext_Model(model_name = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320", num_classes = 500, pretrained = True)

    model.to(config.device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr
    )

    #loss_fn = CrossEntropyLoss()
    loss_fn = FocalLoss()

    trainer = Trainer(
        model=model,
        device=config.device,
        train_dataset=train_dataset,  # 전체 학습 데이터셋
        val_dataset=train_dataset,  # 검증용으로도 동일한 전체 학습 데이터셋 사용
        optimizer=optimizer,
        scheduler=1,
        loss_fn=loss_fn,
        epochs=5,
        result_path=config.save_result_path,
        n_splits=5  # K-Fold의 K 값, 예를 들어 5로 설정
        )

    trainer.train_with_cv()

def test():
    test_info = pd.read_csv(config.test_data_info_file_path)

    test_transform = ConvnextTransform(is_train=False)

    test_dataset = CustomDataset(config.test_data_dir_path,
                                  test_info,
                                  test_transform,
                                  is_inference=True)
    
    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=config.test_shuffle,
                                 drop_last=False)
    
    model = Convnext_Model(model_name = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320", num_classes = 500, pretrained = True)

    model.load_state_dict(
        load_model(config.save_result_path, "best_model.pt")
    )

    predictions = inference_convnext(model, 
                            config.device, 
                            test_loader)

    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("output.csv", index=False)

if __name__ == "__main__":
    main()
    test()