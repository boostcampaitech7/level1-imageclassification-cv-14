import pandas as pd
import torch.optim as optim
import torch
import os
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.base_config import config
from utils.data_related import data_split, get_dataloader
from transforms.convnext_transform import ConvnextTransform
from transforms.sketch_transform_develop import SketchTransform
from dataset.dataset import CustomDataset
from models.convnext_model import Convnext_Model
from losses.cross_entropy_loss import CrossEntropyLoss
from trainers.cv_trainer import Trainer
from utils.inference import load_model, inference_convnext
from losses.Focal_Loss import FocalLoss


def main():
    train_info = pd.read_csv(config.train_data_info_file_path)

    train_transform = SketchTransform(is_train=True)

    train_dataset = CustomDataset(config.train_data_dir_path,
                                  train_info,
                                  train_transform)
    
    model = Convnext_Model(model_name = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320", num_classes = 500, pretrained = True)

    model.to(config.device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr
    )

    loss_fn = CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        device=config.device,
        train_dataset=train_dataset,  # 전체 학습 데이터셋
        val_dataset=None,  # 검증용으로도 동일한 전체 학습 데이터셋 사용
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        epochs=10,
        result_path=config.save_result_path,
        n_splits=5  # K-Fold의 K 값, 예를 들어 5로 설정
        )

    trainer.train_with_cv()


def ensemble_inference(models, device, test_loader):
    """
    여러 모델을 사용하여 앙상블 예측을 수행하는 함수.
    :param models: 불러온 모델들의 리스트.
    :param device: 사용할 장치 (CPU 또는 GPU).
    :param test_loader: 테스트 데이터 로더.
    :return: 앙상블 예측 결과.
    """
    # 모델을 평가 모드로 설정
    for model in models:
        model.to(device)
        model.eval()

    all_predictions = []

    # 각 모델에서의 예측 수행
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Ensembling"):
            if len(images.shape) == 5:  # (num_channels, height, width) 형식인 경우
                images = images.squeeze(1)
            images = images.to(device)
            
            # 각 모델에 대해 예측 수행
            model_outputs = []
            for model in models:
                logits = model(images)
                probs = F.softmax(logits, dim=1)  # 확률로 변환
                model_outputs.append(probs.cpu().numpy())  # 각 모델의 예측을 numpy로 변환
            
            # 모델들의 예측값 평균 (soft voting)
            avg_output = np.mean(model_outputs, axis=0)  # 모델들의 예측 평균
            all_predictions.append(avg_output)

    # 최종 예측 클래스는 평균 확률에서 가장 높은 것을 선택
    final_predictions = np.argmax(np.vstack(all_predictions), axis=1)
    
    return final_predictions

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
    
    # 각 fold에서 저장된 모델 경로
    model_paths = [
        os.path.join(config.save_result_path, f'fold_{i}_best_model.pt') for i in range(5)
    ]

    # 각 fold에서 저장된 모델을 불러와 리스트에 추가
    models = []
    for path in model_paths:
        model = Convnext_Model(model_name="convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320", num_classes=500, pretrained=False)
        model.load_state_dict(load_model(config.save_result_path, os.path.basename(path)))
        models.append(model)
    
    # 장치 설정 (GPU 사용 가능 시 GPU 사용)
    device = config.device
    
    # 앙상블 예측 수행
    predictions = ensemble_inference(models, device, test_loader)

    # 결과 저장
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("output_ensemble.csv", index=False)

if __name__ == "__main__":
    main()
    test()