import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from transforms.sketch_transform_develop import SketchTransform
from dataset.dataset import CustomDataset

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
        n_splits: int = 5  # K-Fold의 K 값, 기본값은 5
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
            val_subset = CustomDataset(self.val_dataset.root_dir, 
                                    info_df.iloc[val_idx].reset_index(drop=True),
                                    transform=val_transform)  # 검증용 Transform 적용

        # 데이터 로더 생성
            train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

            self.model.load_state_dict(self.model.state_dict())  # 사전 학습된 가중치 사용

            # 옵티마이저 초기화
            self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=0.001,
                    weight_decay=1e-4
                )

            # 스케줄러 초기화: StepLR 스케줄러를 사용하여 학습률 조정
            # 스케줄러 초기화
            scheduler_step_size = 30  # 매 30step마다 학습률 감소
            scheduler_gamma = 0.1  # 학습률을 현재의 10%로 감소

            # 한 epoch당 step 수 계산
            steps_per_epoch = len(train_loader)

            # 2 epoch마다 학습률을 감소시키는 스케줄러 선언
            epochs_per_lr_decay = 2
            scheduler_step_size = steps_per_epoch * epochs_per_lr_decay

            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_step_size,
                gamma=scheduler_gamma
)

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

                self.save_model(epoch, val_loss, val_accuracy + train_accuracy)
                
                # 학습률 스케줄러 업데이트
                self.scheduler.step()

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

    def save_model(self, epoch, loss, acc):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}, Acc = {(acc/2):.4f}")