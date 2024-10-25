import os

import torch
import torch.nn as nn
import torch.optim as optim

from configs.base_config import config


from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from transforms.sketch_transform_develop import SketchTransform
from dataset.dataset import CustomDataset

from models.convnext_model import Convnext_Model


import time
import numpy as np
import pandas as pd

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        train_dataset,  # 전체 데이터셋을 입력으로 받음
        val_dataset,  # 검증 데이터셋을 입력으로 받음
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: torch.nn.modules.loss._Loss,
        epochs: int,
        result_path: str,

        n_splits: int = 5,  # K-Fold의 K 값, 기본값은 5

    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result_path = result_path
        self.n_splits = n_splits  # K-Fold의 K 값
        self.best_models = []
        self.lowest_loss = float('inf')

        self.fold_best_models = []


    def train_with_cv(self):
        # StratifiedKFold를 사용한 교차 검증 학습
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # 타겟 레이블을 추출
        targets = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]

        for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(targets)), y=targets)):
            print(f"Fold {fold + 1}/{self.n_splits}")
            train_transform = SketchTransform(is_train=True)
            val_transform = SketchTransform(is_train=False)

             # info_df를 외부에서 전달
            info_df = pd.DataFrame({'image_path': self.train_dataset.image_paths, 'target': self.train_dataset.targets})

            # 각 Fold마다 훈련과 검증 데이터를 위한 데이터셋 생성
            train_subset = CustomDataset(self.train_dataset.root_dir,
                                        info_df.iloc[train_idx].reset_index(drop=True),
                                        transform=train_transform)  # 훈련용 Transform 적용

            val_subset = CustomDataset(self.train_dataset.root_dir, 

                                    info_df.iloc[val_idx].reset_index(drop=True),
                                    transform=val_transform)  # 검증용 Transform 적용

        # 데이터 로더 생성

            train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers = 4)
            val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers = 4)

            self.model = Convnext_Model(model_name = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320", num_classes = 500, pretrained = True)
            self.model.to(config.device)


            # 옵티마이저 초기화
            self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=0.001,
                    weight_decay=1e-4
                )

            # 스케줄러 초기화: StepLR 스케줄러를 사용하여 학습률 조정
            # 스케줄러 초기화

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',  # 손실이 감소하지 않으면 학습률을 줄임
                factor=0.1,  # 학습률 감소 비율
                patience=2,  # 성능 향상이 없을 경우 2 epoch 후에 학습률 감소
                verbose=True
            )


            best_loss = float('inf')
            best_model_path = None


            # 각 fold의 학습 및 검증
            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}")

                start_time = time.time()

                # 올바른 데이터 로더 전달
                train_loss, train_accuracy = self.train_epoch(train_loader)
                val_loss, val_accuracy = self.validate(val_loader)

                end_time = time.time()
                epoch_time = end_time - start_time

                print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                print(f"Epoch {epoch + 1}, Train Acc: {train_accuracy:.4f}, Validation Acc: {val_accuracy:.4f}")
                print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds\n")


                # 저장할 모델 경로 정의
                model_path = os.path.join(self.result_path, f'fold_{fold}_best_model.pt')

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_path = model_path  # 최적 모델 경로 저장
                    self.save_model(model_path)
                    print(f"Best model for Fold {fold + 1} saved at {best_model_path}: Loss: {val_loss:.4f}")

                # 학습률 스케줄러 업데이트
                self.scheduler.step(val_loss)

            # 최적의 모델 경로를 Fold별로 저장
            if best_model_path:
                self.fold_best_models.append(best_model_path)
                print(f"Best model for Fold {fold + 1} saved at {best_model_path}")

    def save_model(self, model_path):
        """
        모델을 주어진 경로에 저장하는 함수
        """
        os.makedirs(self.result_path, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)


    def train_epoch(self, train_loader) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, targets in progress_bar:
            if len(images.shape) == 5:  # (num_channels, height, width) 형식인 경우
                images = images.squeeze(1)
            images = images.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # 가장 높은 값의 인덱스를 예측값으로 선택
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)

            progress_bar.set_postfix(loss=loss.item())

        average_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions * 100  # 퍼센트로 변환

        return [average_loss, accuracy]

    def validate(self, val_loader) -> float:
        # 모델의 검증을 진행
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, targets in progress_bar:
                if len(images.shape) == 5:  # (num_channels, height, width) 형식인 경우
                    images = images.squeeze(1)
                images = images.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  # 가장 높은 값의 인덱스를 예측값으로 선택
                correct_predictions += (predicted == targets).sum().item()
                total_predictions += targets.size(0)

                progress_bar.set_postfix(loss=loss.item())
        # 최종 loss 및 accuracy 계산
        average_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions * 100  # 퍼센트로 변환
        

        return [average_loss, accuracy]

