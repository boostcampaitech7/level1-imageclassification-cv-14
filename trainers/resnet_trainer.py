import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from utils.TimeDecorator import TimeDecorator


class ResTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model  # 훈련할 모델
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수
        self.result_path = result_path  # 모델 저장 경로
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf') # 가장 낮은 Loss를 저장할 변수
        self.scaler = GradScaler()

    def save_model(self, epoch, loss, fold = None):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        if fold is not None:
            current_model_path = os.path.join(self.result_path, f'model_fold_{fold}_epoch_{epoch + 1}_loss_{loss:.4f}.pt')  
            list_len = 0
            best_model_path = f'{fold}fold_best_model.pt'
        else:  
            current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch + 1}_loss_{loss:.4f}.pt')
            list_len = 3
            best_model_path = 'best_model.pt'

        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > list_len:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장

        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, best_model_path)
            torch.save(self.model.state_dict(), best_model_path)
            if fold is not None:
                print(f"Save {fold}fold {epoch + 1}epoch result. Loss = {loss:.4f}")
            else:
                print(f"Save {epoch + 1}epoch result. Loss = {loss:.4f}")

    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        total_loss = 0.0
        correct_pred = 0
        total_pred = 0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            with autocast(device_type=self.device):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            # loss.backward()
            self.scaler.scale(loss).backward()
            # self.optimizer.step()
            self.scaler.step(self.optimizer)

            self.scaler.update()

            self.scheduler.step()
            total_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            correct_pred += (pred == targets).sum().item()
            total_pred += targets.size(0)
            
            progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader), correct_pred / total_pred * 100

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()
        
        total_loss = 0.0
        correct_pred = 0
        total_pred = 0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)    
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                correct_pred += (pred == targets).sum().item()
                total_pred += targets.size(0)

                progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.val_loader), correct_pred / total_pred * 100

    @TimeDecorator()
    def train(self, fold = None) -> None:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            print(f"Epoch {epoch + 1}, Train Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}")

            self.save_model(epoch, val_loss, fold)
            self.scheduler.step()